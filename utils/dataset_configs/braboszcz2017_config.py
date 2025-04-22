from utils.dataset_config import DatasetConfig
from utils.helpers import format_numbers


# =============================================================================
#               Braboszcz et al. (2017) EEG Dataset Configuration
# =============================================================================

subjects = []
subjects.extend(range(1, 25))
subjects.extend(range(25, 56))
subjects.extend(range(56, 60))
subjects.extend(range(60, 79))
subjects.extend(range(79, 99))
subjects = format_numbers(subjects,3)  # convert to list of strings with leading zeros like ["060", "061", ...]
task_orientation = "internal"

# Initialize an empty dictionary for subject grouping
subject_groups = {}

# Add subjects to the dictionary based on their group
for i in range(1, 25):
    subject_groups[format_numbers(i, 3)] = "htr" #Himalayan

for i in range(25, 56):
    subject_groups[format_numbers(i, 3)] = "ctr" #Control

for i in range(56, 60):
    subject_groups[format_numbers(i, 3)] = "tm" #Transcendental???

for i in range(60, 79):
    subject_groups[format_numbers(i, 3)] = "vip" #Vipassana

for i in range(79, 99):
    subject_groups[format_numbers(i, 3)] = "sny" #Isha Shoonya Yoga


# Creating event id for each task and practice
event_id = {'htr/med1':1, 'htr/med2':2, 'htr/think1':3, 'htr/think2':4,
          'ctr/med1':5, 'ctr/med2':6, 'ctr/think1':7, 'ctr/think2':8,
          'tm/med1': 9, 'tm/med2' :10, 'tm/think1':11, 'tm/think2':12,
          'vip/med1':13, 'vip/med2':14, 'vip/think1':15, 'vip/think2':16,
          'sny/med1':17, 'sny/med2':18, 'sny/think1':19, 'sny/think2':20}

DATASET_CONFIG = DatasetConfig(
    name="Braboszcz et al. (2017)",
    f_name="braboszcz2017",
    extension="bdf",
    task_orientation=task_orientation,
    subjects=subjects,
    tasks=["med2", "think2"], 
    mapping_channels={},  
    mapping_non_eeg={},   
    event_id_map=event_id,
    state_classes={
        "focused": [event_id['htr/med1'], event_id['htr/med2'],
                    event_id['ctr/med1'], event_id['ctr/med2'],
                    event_id['tm/med1'], event_id['tm/med2'],
                    event_id['vip/med1'], event_id['vip/med2'],
                    event_id['sny/med1'], event_id['sny/med2']],
        "mind-wandering": [event_id['htr/think1'], event_id['htr/think2'],
                           event_id['ctr/think1'], event_id['ctr/think2'],
                           event_id['tm/think1'], event_id['tm/think2'],
                           event_id['vip/think1'], event_id['vip/think2'],
                           event_id['sny/think1'], event_id['sny/think2']],
    },
    task_classes={
        "breath_counting": [event_id['htr/med1'], event_id['ctr/med1'],
                            event_id['tm/med1'], event_id['vip/med1'],
                            event_id['sny/med1'], event_id['ctr/med2']], # control group performed breath counting in both med1 and med2
        "meditation": [event_id['htr/med2'], event_id['tm/med2'], 
                       event_id['vip/med2'], event_id['sny/med2']],
        "thinking": [event_id['htr/think1'], event_id['htr/think2'],
                     event_id['ctr/think1'], event_id['ctr/think2'],
                     event_id['tm/think1'], event_id['tm/think2'],
                     event_id['vip/think1'], event_id['vip/think2'],
                     event_id['sny/think1'], event_id['sny/think2']],
    },
    extra_info={
        "subject_groups": subject_groups
    }
)
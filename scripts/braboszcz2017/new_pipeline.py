import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from autoreject import AutoReject, Ransac, get_rejection_threshold

import utils.config as config
from utils.config import DATASETS
from utils.file_io import save_epochs
from utils.helpers import format_numbers

### Defining constants and preparing stuff ###

DATASET = DATASETS["braboszcz2017"]

bids_root = DATASET.path_raw
path_epochs = os.path.join(DATASET.path_epochs, "preprocessed")
os.makedirs(path_epochs, exist_ok=True)

# EEG settings
subjects = DATASET.subjects
sessions = DATASET.sessions
tasks = ["med2", "think2"]
event_id = DATASET.event_id_map
subject_groups = DATASET.extra_info["subject_groups"]
epoch_length = config.EEG_SETTINGS["EPOCH_LENGTH_SEC"] 
tmin = config.EEG_SETTINGS["EPOCH_START_SEC"]
tmax = tmin + epoch_length
sfreq = config.EEG_SETTINGS["SAMPLING_RATE"]
h_cut = config.EEG_SETTINGS["HIGH_CUTOFF_HZ"]
l_cut = config.EEG_SETTINGS["LOW_CUTOFF_HZ"]
reject_threshold = config.EEG_SETTINGS["REJECT_THRESHOLD"]
montage = mne.channels.make_standard_montage('biosemi64')




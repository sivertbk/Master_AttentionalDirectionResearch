import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from joblib import cpu_count
from autoreject import AutoReject, Ransac, get_rejection_threshold
from autoreject.utils import interpolate_bads

import utils.config as config
from utils.config import DATASETS
from utils.file_io import save_epochs, load_raw_data
from utils.helpers import format_numbers

##----------------------------------------------------------------------------##
#                  Defining constants and preparing stuff                      #
##----------------------------------------------------------------------------##

DATASET = DATASETS["jin2019"]

# Paths
bids_root = DATASET.path_raw
path_epochs = os.path.join(DATASET.path_epochs, "preprocessed")
path_derivatives = DATASET.path_derivatives
path_plots = os.path.join(config.PLOTS_PATH, DATASET.f_name)
os.makedirs(path_epochs, exist_ok=True)
os.makedirs(path_derivatives, exist_ok=True)
os.makedirs(path_plots, exist_ok=True)

# EEG settings
subjects = DATASET.subjects
sessions = DATASET.sessions
event_id = DATASET.event_id_map
epoch_length = config.EEG_SETTINGS["EPOCH_LENGTH_SEC"] 
tmin = config.EEG_SETTINGS["EPOCH_START_SEC"]
tmax = tmin + epoch_length
sfreq = config.EEG_SETTINGS["SAMPLING_RATE"]
h_cut = config.EEG_SETTINGS["HIGH_CUTOFF_HZ"]
l_cut = config.EEG_SETTINGS["LOW_CUTOFF_HZ"]
reject_threshold = config.EEG_SETTINGS["REJECT_THRESHOLD"]
montage = mne.channels.make_standard_montage('biosemi64')


# Temporary defs
subject = 17
session = 2
task = None


##----------------------------------------------------------------------------##
#                                   FUNCTIONS                                  #
##----------------------------------------------------------------------------##



##----------------------------------------------------------------------------##
#                         LOAD AND PREPARE RAW EEG DATA                        #
##----------------------------------------------------------------------------##
''' Load the raw data and create copies for further processing. All channel
    types will be set correctly. Data will be downsampled to sampling rate
    defined in config.py.

    RESULTS: 
    - Raw data objects from the respective dataset ready to be preprocessed.
'''

raw = load_raw_data(DATASET, subject, session, task=task, preload=True)
raw.set_channel_types({'EXG1': 'misc',
                    'EXG2': 'misc',
                    'EXG3': 'misc',
                    'EXG4': 'misc',
                    'EXG5': 'misc',
                    'EXG6': 'misc',
                    'EXG7': 'misc',
                    'EXG8': 'misc',
                    'GSR1': 'misc',
                    'GSR2': 'misc',
                    'Erg1': 'misc',
                    'Erg2': 'misc',
                    'Resp': 'bio',
                    'Plet': 'bio',
                    'Temp': 'bio'
})
raw.resample(sfreq)
raw.set_montage(montage)
raw.info['bads'] = []  # Reset bad channels
raw.del_proj()  # Remove proj, don't proj while interpolating

##----------------------------------------------------------------------------##
#              DETECT BAD CHANNELS AUTOMATICALLY USING RANSAC                  #
##----------------------------------------------------------------------------##
''' Create a high-pass filtered copy of the raw data and generate fixed-length
    epochs. Use RANSAC to automatically detect consistently bad channels.

    RESULTS:
    - List of bad channels: ransac.bad_chs_
'''

raw_hp = raw.copy().filter(l_freq=l_cut, h_freq=None) 
epochs_hp = mne.make_fixed_length_epochs(raw_hp, duration=epoch_length, preload=True)
picks = mne.pick_types(epochs_hp.info, meg=False, eeg=True,
                       stim=False, eog=False,
                       include=[], exclude=[])

ransac = Ransac(verbose=True,picks=picks, n_jobs=cpu_count())
ransac.fit(epochs_hp)
bad_chs = ransac.bad_chs_
print(bad_chs)

##----------------------------------------------------------------------------##
#                  INTERPOLATE BAD CHANNELS IN ORIGINAL RAW                    #
##----------------------------------------------------------------------------##
''' Interpolate bad channels in the original raw data based on RANSAC results.

    RESULTS:
    - Cleaned raw object with interpolated channels
'''

# raw.info['bads'] = bad_chs
# raw.interpolate_bads(reset_bads=True)

##----------------------------------------------------------------------------##
#           CREATE NEW SYNTHETIC EPOCHS AND DETECT BAD EPOCHS (AR)             #
##----------------------------------------------------------------------------##
''' Create new synthetic epochs from the interpolated raw data using linear
    detrending (no high-pass filter), and apply AutoReject to detect bad epochs.

    RESULTS:
    - Clean epochs
    - Reject log with bad epochs
'''

# epochs_ar = mne.make_fixed_length_epochs(raw.copy(), duration=3.0, detrend=1, preload=True)
# ar = AutoReject(n_jobs=1, random_state=42)
# epochs_ar_clean, reject_log = ar.fit_transform(epochs_ar, return_log=True)

##----------------------------------------------------------------------------##
#                    AVERAGE REFERENCE CLEAN EPOCHS (AR)                       #
##----------------------------------------------------------------------------##
''' Apply average reference to epochs after removing bad epochs.

    RESULTS:
    - Referenced epochs ready for ICA
'''

# epochs_ar_clean.set_eeg_reference('average', projection=False) # proj=False -> reference is directly applied on the data

##----------------------------------------------------------------------------##
#                       FIT ICA ON CLEAN EPOCH DATA                            #
##----------------------------------------------------------------------------##
''' Fit ICA on clean, average-referenced synthetic epochs. Only epochs that
    were marked as good by AutoReject are used.

    RESULTS:
    - ICA object trained on clean EEG data
'''

# ica = ICA(n_components=0.99, random_state=97)
# ica.fit(epochs_ar_clean[~reject_log.bad_epochs])

##----------------------------------------------------------------------------##
#              IDENTIFY ARTIFACT COMPONENTS AUTOMATICALLY (EOG)                #
##----------------------------------------------------------------------------##
''' Automatically identify EOG-related ICA components using correlation with
    EOG channels.

    RESULTS:
    - ICA.exclude list updated with suspected EOG artifacts
'''

# eog_inds, scores = ica.find_bads_eog(epochs_ar_clean, ch_name='EOG061')
# ica.exclude = eog_inds

##----------------------------------------------------------------------------##
#             APPLY ICA TO FULL UNFILTERED RAW (CLEANED VERSION)               #
##----------------------------------------------------------------------------##
''' Apply the ICA solution (trained on clean synthetic epochs) to the full,
    unfiltered raw data. This ensures all data benefits from artifact removal.

    RESULTS:
    - ICA-cleaned raw object
'''

# ica.apply(raw)

##----------------------------------------------------------------------------##
#         EPOCH THE CLEAN RAW FOR ANALYSIS AND APPLY DETRENDING                #
##----------------------------------------------------------------------------##
''' Epoch the ICA-cleaned raw based on experimental events and apply linear
    detrending to preserve frequency information while removing slow drifts.

    RESULTS:
    - Analysis-ready epochs
'''

# epochs = mne.Epochs(raw, events, tmin=-5.0, tmax=0.0, detrend=1, preload=True)

##----------------------------------------------------------------------------##
#        FINAL CLEANING OF ANALYSIS EPOCHS USING AUTOREJECT (FINAL AR)         #
##----------------------------------------------------------------------------##
''' Apply AutoReject to final analysis epochs to remove residual transients
    and improve spectral data quality.

    RESULTS:
    - Final clean epochs ready for frequency analysis
    - Final reject log
'''

# ar_final = AutoReject(n_jobs=1, random_state=42)
# epochs_final, reject_log_final = ar_final.fit_transform(epochs, return_log=True)

##----------------------------------------------------------------------------##
#         SAVE FINAL OUTPUTS, LOGS, FIGURES, AND PROCESSING REPORTS            #
##----------------------------------------------------------------------------##
''' Save the cleaned epochs, bad channel lists, ICA exclusion info, reject logs,
    and preprocessing figures to dedicated output folders for traceability.

    RESULTS:
    - epochs_final.fif, ICA components, bad_channels.json, reject_logs.csv, etc.
'''




# epochs_final.save(...)
# ica.save(...)
# save bad_chs to JSON / CSV
# save reject logs, ICA plots, bad epoch figures, etc.


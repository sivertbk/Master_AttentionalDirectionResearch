import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from joblib import cpu_count
from autoreject import AutoReject, Ransac, read_auto_reject
from autoreject.utils import set_matplotlib_defaults
from scipy.signal import detrend
from mne.preprocessing import ICA

import utils.config as config
from utils.config import DATASETS, set_plot_style
from utils.file_io import load_raw_data, save_epochs, load_bad_channels, update_bad_channels_json, save_autoreject, load_reject_log
from utils.helpers import format_numbers, cleanup_memory, plot_ransac_bad_log, reject_epochs_with_adjacent_interpolation, plot_rejected_epochs_by_cluster, plot_dropped_epochs_by_cluster, update_bad_epochs_from_indices

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
mapping_128_to_64 = DATASET.mapping_channels
mapping_non_eeg = DATASET.mapping_non_eeg

set_plot_style()

# Run options
USE_CACHED_RANSAC = True           # Load RANSAC bad channels from JSON if available
USE_CACHED_PRE_AUTOREJECT = True   # Load autoreject log (.npz) and ar model if available
USE_CACHED_POST_AUTOREJECT = True  # Load autoreject log (.npz) if available

SAVE_PLOTS = True                  # Whether to save generated plots
SHOW_PLOTS = True                  # Whether to show generated plots
VERBOSE = True                     # Toggle verbose outputs/logs
DEBUG = False                      # Toggle debug plots, cluster checks, etc.

# Temporary defs
subject = 8
session = 2
task = None
run = None


##----------------------------------------------------------------------------##
#                    1. LOAD AND PREPARE RAW EEG DATA                          #
##----------------------------------------------------------------------------##
''' Load the raw data and create copies for further processing. All channel
    types will be set correctly. Data will be downsampled to sampling rate
    defined in config.py.

    RESULTS: 
    - Raw data objects from the respective dataset ready to be preprocessed.
'''

raw = load_raw_data(DATASET, subject, session, task=task, preload=True)
# Extracting the channel names
old_ch_names = raw.ch_names

### Temporary channel names ###

# Placeholder names for the old channels
temp_ch_names = ['temp_' + ch for ch in old_ch_names[:-9]]
temp_ch_names.extend(old_ch_names[-9:])
mapping_old_to_temp = dict(zip(old_ch_names, temp_ch_names))

# Rename the channels in the dataset
raw.rename_channels(mapping_old_to_temp)
raw.rename_channels(mapping_128_to_64)
raw.rename_channels(mapping_non_eeg)

# Set the channel types for the EXG channels
raw.set_channel_types({
    'sacc_EOG1': 'eog',
    'sacc_EOG2': 'eog',
    'blink_EOG1': 'eog',
    'blink_EOG2': 'eog',
    'EXG5': 'misc',  # Could be a mastoid, set as misc
    'EXG6': 'misc',  # Could be a mastoid, set as misc
    'EXG7': 'misc',  # Could be a mastoid, set as misc
    'EXG8': 'misc'   # Could be a mastoid, set as misc
})

# Identify non-EEG channels
non_eeg_channels = [ch for ch, ch_type in zip(raw.ch_names, raw.get_channel_types()) if ch_type != 'eeg']

# Get a list of the channels you want to retain (the new 64-channel names and the non-EEG channels)
channels_to_keep = list(mapping_128_to_64.values()) + non_eeg_channels

# Drop the channels not in the 64-channel system
raw.pick(channels_to_keep)

raw.resample(sfreq)
raw.set_montage(montage)
raw.info['bads'] = []  # Reset bad channels
raw.del_proj()  # Remove proj, don't proj while interpolating

# raw.plot(highpass=1, lowpass=None, block=True)

##----------------------------------------------------------------------------##
#            2. DETECT BAD CHANNELS AUTOMATICALLY USING RANSAC                 #
##----------------------------------------------------------------------------##
''' Create a high-pass filtered copy of the raw data and generate fixed-length
    epochs. Use RANSAC to automatically detect consistently bad channels.

    RESULTS:
    - List of bad channels: ransac.bad_chs_
'''
if USE_CACHED_RANSAC:
    bad_chs = load_bad_channels(
        save_dir=path_derivatives,
        dataset=DATASET.f_name,
        subject=subject,
        session=session
    )
else:
    bad_chs = None

if bad_chs is None:
    raw_hp = raw.copy().filter(l_freq=l_cut, h_freq=None) 

    epochs_hp = mne.make_fixed_length_epochs(raw_hp, 
                                            duration=epoch_length, 
                                            preload=True,
                                            verbose=VERBOSE)

    picks_eeg = mne.pick_types(epochs_hp.info, meg=False, eeg=True,
                        stim=False, eog=False,
                        include=[], exclude=[])

    ransac = Ransac(
        picks=picks_eeg,
        n_resample=100,       # Keep default
        min_channels=0.15,    # Using less channels than defualt
        min_corr=0.80,       # Accept lower correlation → fewer false positives
        unbroken_time=0.25,   # Allow sensors to be bad more of the time before flagging
        n_jobs=cpu_count(),  # Use all available CPU cores
        random_state=435656,
        verbose=VERBOSE
    )
    ransac.fit(epochs_hp)
    bad_chs = ransac.bad_chs_

    update_bad_channels_json(
        save_dir=path_derivatives,
        bad_chs=bad_chs,
        subject=subject,
        dataset=DATASET.f_name,
        session=session
    )

    plot_ransac_bad_log(
        ransac=ransac,
        epochs_hp=epochs_hp,
        subject_id=subject,
        dataset=DATASET.name,
        meta_info=session,
        save_path=os.path.join(path_plots, "bad_chans", f"sub-{subject}_ses-{session}_bad_chans_ransac.png") if SAVE_PLOTS else None,
        show=SHOW_PLOTS 
    )



##----------------------------------------------------------------------------##
#               3. INTERPOLATE BAD CHANNELS IN ORIGINAL RAW                    #
##----------------------------------------------------------------------------##
''' Interpolate bad channels in the original raw data based on RANSAC results.

    RESULTS:
    - Cleaned raw object with interpolated channels
'''
# plot the bad channels
picks_bads = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, include=bad_chs, exclude=[])
fig = raw.plot(highpass=l_cut, picks=picks_bads, scalings=dict(eeg=100e-6), title=f"Bad channels for subject {subject} - session  {session}", show=SHOW_PLOTS, block=SHOW_PLOTS)
if SAVE_PLOTS:
    fig.savefig(os.path.join(path_plots, "bad_chans", f"sub-{subject}_ses-{session}_bad_chans.png"))
raw.info['bads'] = bad_chs
fig = raw.plot_sensors(show_names=True, kind="topomap", show=SHOW_PLOTS, block=SHOW_PLOTS)
if SAVE_PLOTS:
    fig.savefig(os.path.join(path_plots, "bad_chans", f"sub-{subject}_ses-{session}_sensors.png"))

# Interpolate bad channels
raw.info['bads'] = bad_chs
raw.interpolate_bads(reset_bads=True)

# plot the interpolated channels
picks_bads = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, include=bad_chs, exclude=[])
fig = raw.plot(highpass=l_cut, picks=picks_bads, scalings=dict(eeg=100e-6), title=f"Interpolated channels for subject {subject} - session  {session}", show=SHOW_PLOTS, block=SHOW_PLOTS)
if SAVE_PLOTS:
    fig.savefig(os.path.join(path_plots, "bad_chans", f"sub-{subject}_ses-{session}_interpolated_chans.png"))

##----------------------------------------------------------------------------##
#        4. CREATE NEW SYNTHETIC EPOCHS AND DETECT BAD EPOCHS (AR)             #
##----------------------------------------------------------------------------##
''' Create new synthetic epochs from the interpolated raw data using linear
    detrending (no high-pass filter), baseline correction, and then apply 
    AutoReject to detect bad epochs.

    RESULTS:
    - Clean epochs
    - Reject log with bad epochs
'''


epochs_to_ica = mne.make_fixed_length_epochs(
    raw.copy().filter(l_freq=l_cut, h_freq=None), 
    duration=epoch_length, 
    preload=True)

picks_ica = mne.pick_types(
    epochs_to_ica.info, 
    meg=False, eeg=True,
    stim=False, eog=True,
    include=[], exclude=[])

picks_eeg = mne.pick_types(
    epochs_to_ica.info, 
    meg=False, eeg=True,
    stim=False, eog=False,
    include=[], exclude=[])

epochs_to_ar = epochs_to_ica.copy().pick(picks_eeg) 
epochs_to_ica.pick(picks_ica)

# Apply detrending
#epochs_to_ica._data = detrend(epochs_to_ica.get_data(), axis=2, type='linear')

# Apply baseline correction
#epochs_to_ica = epochs_to_ica.apply_baseline((None, None))  # from start to end of epoch


if USE_CACHED_PRE_AUTOREJECT:
    # Load the autoreject object
    ar = read_auto_reject(
        os.path.join(path_derivatives, "pre_ica_autoreject_objects", f"sub-{subject}_ses-{session}_autoreject_pre.h5")
    )
    # Check if the autoreject object is None
    if ar is None:
        print(f"AutoReject object not found for subject {subject}.")

    # Load the autoreject log
    reject_log = load_reject_log(
    path=os.path.join(path_derivatives, "pre_ica_autoreject_logs"),
    subject=subject,
    session=session
    )

    # if reject_log is not None:
    #     print(f"Reject log not found for subject {subject}.")
        
    #     # Reject epochs with clustered interpolation
    #     epochs_ar_cleaned, rejected_idxs = reject_epochs_with_adjacent_interpolation(
    #         epochs_to_ica, 
    #         reject_log, 
    #         max_cluster_size=3,
    #         verbose=VERBOSE
    #     )

    #     print(f"[Cluster Filter] Rejected {len(rejected_idxs)} out of {len(epochs_to_ica)} epochs due to clustered interpolation")
    #     # Update the reject log with the new bad epochs
    #     print(reject_log.bad_epochs)
    #     update_bad_epochs_from_indices(
    #         reject_log,
    #         rejected_idxs,
    #         verbose=VERBOSE
    #     )
    #     print(reject_log.bad_epochs)
        
else:
    reject_log = None

if reject_log is None:
    ar = AutoReject(
        n_interpolate=np.array([2, 8, 12]),     # Try several interpolation levels
        consensus=np.linspace(0.3, 0.7, 7),        # Require some agreement, not too harsh
        thresh_method='bayesian_optimization',
        cv=10,                                     # cross validation: K-fold
        picks=picks_eeg,
        n_jobs=cpu_count(),  # Use all available CPU cores
        random_state=42,
        verbose=True
    )

    # Fit AutoReject to the epochs
    ar.fit(epochs_to_ar[:min(len(epochs_to_ar),400)])

    # transform epochs and get the reject log
    epochs_ar, reject_log = ar.transform(epochs_to_ar.copy(), return_log=True)

    # Reject epochs with clustered interpolation
    # epochs_ar_cleaned, rejected_idxs = reject_epochs_with_adjacent_interpolation(
    #     epochs_to_ar, 
    #     reject_log, 
    #     max_cluster_size=3,
    #     verbose=VERBOSE
    # )

    # print(f"[Cluster Filter] Rejected {len(rejected_idxs)} out of {len(epochs_to_ar)} epochs due to clustered interpolation")

    # # Update the reject log with the new bad epochs
    # update_bad_epochs_from_indices(
    #     reject_log,
    #     rejected_idxs,
    #     verbose=VERBOSE
    # )


    #######     SAVING, PLOTTING, AND LOGGING STUFF      #######

    # if SHOW_PLOTS:
    #     plot_dropped_epochs_by_cluster(
    #         epochs=epochs_to_ar,
    #         reject_log=reject_log,
    #         rejected_indices=rejected_idxs,
    #         title="Rejected due to spatially adjacent interpolations"
    #     )

    fig_ar_epochs_before_ica = epochs_to_ar[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6), show=SHOW_PLOTS, block=SHOW_PLOTS)
    fig_ar_matrix_before_ica = reject_log.plot(orientation="horizontal", show=SHOW_PLOTS)
    if SAVE_PLOTS:
        fig_ar_matrix_before_ica.axes[0].set_title(f"AutoReject bad/interpolated epochs pre ICA | Dataset: {DATASET.name} | Subject: {subject} | {task}.")
        save_path = os.path.join(path_plots, "bad_epochs_pre_ICA_matrix")
        os.makedirs(save_path, exist_ok=True)
        fig_ar_matrix_before_ica.savefig(os.path.join(save_path , f"sub-{subject}_ses-{session}_bad_epochs_pre_ICA_matrix.png"))

    # save ar to derivatives
    save_path = os.path.join(path_derivatives, "pre_ica_autoreject_objects")
    os.makedirs(save_path, exist_ok=True)
    save_autoreject(ar, os.path.join(save_path, f"sub-{subject}_ses-{session}_autoreject_pre.h5"))

    # save reject log to derivatives
    save_path = os.path.join(path_derivatives, "pre_ica_autoreject_logs")
    os.makedirs(save_path, exist_ok=True)
    reject_log.save(os.path.join(save_path, f"sub-{subject}_ses-{session}_autoreject_log.npz"), overwrite=True)


##----------------------------------------------------------------------------##
#                 5. AVERAGE REFERENCE CLEAN EPOCHS (AR)                       #
##----------------------------------------------------------------------------##
''' Apply average reference to epochs after removing bad epochs.

    RESULTS:
    - Referenced epochs ready for ICA
'''

epochs_to_ica.set_eeg_reference('average', verbose=VERBOSE, projection=False) # proj=False -> reference is directly applied on the data

##----------------------------------------------------------------------------##
#                    6. FIT ICA ON CLEAN EPOCH DATA                            #
##----------------------------------------------------------------------------##
''' Fit ICA on clean, average-referenced synthetic epochs. Only epochs that
    were marked as good by AutoReject are used.

    RESULTS:
    - ICA object trained on clean EEG data
''' 

# remove existing projections
projs = epochs_to_ica.info['projs']
epochs_to_ica.del_proj()

ica = ICA(method="infomax", 
          n_components=None, 
          random_state=97,
          verbose=VERBOSE)


ica.fit(epochs_to_ica[~reject_log.bad_epochs], picks=picks_ica, verbose=VERBOSE)
ica.plot_components(title=f"ICA decomposition on subject {subject}")
ica.plot_sources(epochs_to_ica, block=True) 

##----------------------------------------------------------------------------##
#           7. IDENTIFY ARTIFACT COMPONENTS AUTOMATICALLY (EOG)                #
##----------------------------------------------------------------------------##
''' Automatically identify EOG-related ICA components using correlation with
    EOG channels.

    RESULTS:
    - ICA.exclude list updated with suspected EOG artifacts
'''

eog_inds, scores = ica.find_bads_eog(epochs_to_ica, threshold=3.0)
ica.exclude = eog_inds

##----------------------------------------------------------------------------##
#          8. APPLY ICA TO FULL UNFILTERED RAW (CLEANED VERSION)               #
##----------------------------------------------------------------------------##
''' Apply the ICA solution (trained on clean synthetic epochs) to the full,
    unfiltered raw data. This ensures all data benefits from artifact removal.

    RESULTS:
    - ICA-cleaned raw object
'''
# Set average reference to original raw before applying ICA
raw.set_eeg_reference('average', projection=False)  
raw_from_ica = ica.apply(raw.copy())

ica.plot_overlay(epochs_to_ica.average(), exclude=ica.exclude)
ica.apply(epochs_to_ica, exclude=ica.exclude)

# Plot cleaned data against bad data (averaged)
evoked_bad = epochs_to_ica[reject_log.bad_epochs].average()
plt.figure()
plt.plot(evoked_bad.times, evoked_bad.data.T * 1e6, 'r', zorder=-1)
epochs_ar.average().plot(axes=plt.gca())

##----------------------------------------------------------------------------##
#      9. EPOCH THE CLEAN RAW FOR ANALYSIS AND APPLY DETRENDING                #
##----------------------------------------------------------------------------##
''' Epoch the ICA-cleaned raw based on experimental events and apply linear
    detrending to preserve frequency information while removing slow drifts.

    RESULTS:
    - Analysis-ready epochs
'''

# epochs = mne.Epochs(raw, events, tmin=-5.0, tmax=0.0, detrend=1, preload=True)

##----------------------------------------------------------------------------##
#      10. FINAL CLEANING OF ANALYSIS EPOCHS USING AUTOREJECT (FINAL AR)       #
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
#       11. SAVE FINAL OUTPUTS, LOGS, FIGURES, AND PROCESSING REPORTS          #
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


cleanup_memory('raw', 'raw_hp', 'epochs_hp', 'epochs_ar', 'epochs_ar_cleaned', 'reject_log', 'reject_log_final', 'ica', 'ar', 'ar_final')

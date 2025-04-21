import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import psutil
import gc
from joblib import cpu_count
from autoreject import AutoReject, Ransac, get_rejection_threshold
from autoreject.utils import set_matplotlib_defaults
from scipy.signal import detrend
from mne.preprocessing import ICA

import utils.config as config
from utils.config import DATASETS, set_plot_style
from utils.file_io import load_raw_data, save_epochs, load_bad_channels, update_bad_channels_json, save_autoreject, load_reject_log
from utils.helpers import format_numbers, cleanup_memory, plot_ransac_bad_log, reject_epochs_with_adjacent_interpolation, plot_rejected_epochs_by_cluster, plot_dropped_epochs_by_cluster, update_bad_epochs_from_indices


def prepare_raw_data(raw, dataset, eeg_settings):
    """
    Prepare data for preprocessing. setting correct channel types, selecting channels, resample, set montage, and more deping on the dataset.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw data object to prepare.
    dataset : str
        The name of the dataset being processed.
    eeg_settings : dict
        The EEG settings dictionary defined in config containing parameters
    Returns
    -----------
    raw : mne.io.Raw
        The prepared raw data object.
    """
    if dataset.f_name == 'braboszcz2017':
        raw = _prepare_braboszcz2017(raw, dataset, eeg_settings)

    elif dataset.f_name == 'jin2019':
        raw = _prepare_jin2019(raw, dataset, eeg_settings)
    
    elif dataset.f_name == 'touryan2022':
        raw = _prepare_touryan2022(raw, dataset, eeg_settings)
    else:
        raise ValueError(f"Unknown dataset: {dataset.f_name}")
    
    return raw


def _prepare_braboszcz2017(raw, dataset, eeg_settings):
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
    raw.resample(eeg_settings["SAMPLING_RATE"])
    raw.set_montage(eeg_settings["MONTAGE"])
    raw.info['bads'] = []  # Reset bad channels
    raw.del_proj()  # Remove proj, don't proj while interpolating

    return raw

def _prepare_jin2019(raw, dataset, eeg_settings):
    # Extracting the channel names
    old_ch_names = raw.ch_names

    ### Temporary channel names ###

    # Placeholder names for the old channels
    temp_ch_names = ['temp_' + ch for ch in old_ch_names[:-9]]
    temp_ch_names.extend(old_ch_names[-9:])
    mapping_old_to_temp = dict(zip(old_ch_names, temp_ch_names))

    # Rename the channels in the dataset
    raw.rename_channels(mapping_old_to_temp)
    raw.rename_channels(dataset.mapping_channels)
    raw.rename_channels(dataset.mapping_non_eeg)

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
    channels_to_keep = list(dataset.mapping_channels.values()) + non_eeg_channels

    # Drop the channels not in the 64-channel system
    raw.pick(channels_to_keep)
    raw.resample(eeg_settings["SAMPLING_RATE"])
    raw.set_montage(eeg_settings["MONTAGE"])
    raw.info['bads'] = []  # Reset bad channels
    raw.del_proj()  # Remove proj, don't proj while interpolating

    return raw

def _prepare_touryan2022(raw, dataset, eeg_settings):
    
    raw.resample(eeg_settings["SAMPLING_RATE"])
    raw.set_montage(eeg_settings["MONTAGE"])
    raw.info['bads'] = []  # Reset bad channels
    raw.del_proj()  # Remove proj, don't proj while interpolating

    return raw

def ransac_detect_bad_channels(raw, dataset, eeg_settings, subject, session=None, task=None, run=None, verbose=True, save_plot=True, show_plot=True):
    raw_hp = raw.copy().filter(l_freq=eeg_settings['LOW_CUTOFF_HZ'], h_freq=None)

    epochs_hp = mne.make_fixed_length_epochs(raw_hp, 
                                            duration=eeg_settings["EPOCH_LENGTH_SEC"], 
                                            preload=True,
                                            verbose=verbose)
    
    picks_eeg = mne.pick_types(epochs_hp.info, meg=False, eeg=True,
                        stim=False, eog=False,
                        include=[], exclude=[])
    
    ransac = Ransac(
        picks=picks_eeg,
        n_resample=101,       # Double the number of epochs to resample and uneven number to get true median
        min_channels=0.0625,  # Using less channels than defualt (4 channels)
        min_corr=0.60,        # Accept lower correlation → fewer false positives
        unbroken_time=0.25,   # Allow sensors to be bad more of the time before flagging
        n_jobs=cpu_count(),   # Use all available CPU cores
        random_state=42069,   # For reproducibility
        verbose=verbose
    )

    ransac.fit(epochs_hp)
    bad_chs = ransac.bad_chs_

    update_bad_channels_json(
    save_dir=dataset.path_derivatives,
    bad_chs=bad_chs,
    subject=subject,
    dataset=dataset.f_name, 
    session=session,
    task=task,
    run=run,
    )

    path_plots = os.path.join(config.PLOTS_PATH, dataset.f_name)
    os.makedirs(path_plots, exist_ok=True)

    # Choose the appropriate meta_info value
    if session is not None:
        meta_info = session
    elif task is not None:
        meta_info = task
    elif run is not None:
        meta_info = run
    else:
        meta_info = "Unknown"  # Or set to a default value if none of the above is available

    plot_ransac_bad_log(
        ransac=ransac,
        epochs_hp=epochs_hp,
        subject_id=subject,
        dataset=dataset.name,
        meta_info=meta_info,
        save_path=os.path.join(path_plots, "bad_chans", f"sub-{subject}_ses-{meta_info}_bad_chans_ransac.png") if save_plot else None,
        show=show_plot 
    )

    # plot the bad channels
    raw.info['bads'] = bad_chs

    fig = raw.plot_sensors(show_names=True, kind="topomap", show=False)
    if save_plot:
        fig.savefig(os.path.join(path_plots, "bad_chans", f"sub-{subject}_ses-{meta_info}_sensors.png"))


def fix_bad_channels(raw, dataset, eeg_settings, subject, session=None, task=None, run=None):
    """
    Fix bad channels in the raw data. This function loads the bad channels from a JSON file and updates the raw data accordingly.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw data object to fix bad channels.
    dataset : str
        The name of the dataset being processed.
    eeg_settings : dict
        The EEG settings dictionary defined in config containing parameters
    
    Returns
    -----------
    raw : mne.io.Raw
        The raw data object with updated bad channels.
    """
    # Load the bad channels from the JSON file
    bad_channels = load_bad_channels(dataset.f_name)

    # Update the raw data with the bad channels
    raw.info['bads'] = bad_channels


    return raw


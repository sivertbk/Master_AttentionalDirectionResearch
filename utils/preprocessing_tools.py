import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
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
        return _prepare_braboszcz2017(raw, dataset, eeg_settings)
    elif dataset.f_name == 'jin2019':
        return _prepare_jin2019(raw, dataset, eeg_settings)
    elif dataset.f_name == 'touryan2022':
        return _prepare_touryan2022(raw, dataset, eeg_settings)
    else:
        raise ValueError(f"Unknown dataset: {dataset.f_name}")
    


def _prepare_braboszcz2017(raw, dataset, eeg_settings):
    # Define desired channel type mappings
    desired_types = {
        'EXG1': 'misc', 'EXG2': 'misc', 'EXG3': 'misc', 'EXG4': 'misc',
        'EXG5': 'misc', 'EXG6': 'misc', 'EXG7': 'misc', 'EXG8': 'misc',
        'GSR1': 'misc', 'GSR2': 'misc',
        'Erg1': 'misc', 'Erg2': 'misc',
        'Resp': 'bio', 'Plet': 'bio', 'Temp': 'bio'
    }

    # Only keep keys that exist in raw
    existing_types = {
        ch: typ for ch, typ in desired_types.items()
        if ch in raw.info['ch_names']
    }

    # Set only the types for present channels
    raw.set_channel_types(existing_types)

    # Downsample
    raw.resample(eeg_settings["SAMPLING_RATE"])

    # Apply EEG montage
    raw.set_montage(eeg_settings["MONTAGE"], on_missing="ignore")

    # Clear projections and bad channels
    raw.info['bads'] = []
    raw.del_proj()

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
    # Set the channel types
    raw.set_channel_types({
        'LHEOG': 'eog',
        'RHEOG': 'eog',
        'UVEOG': 'eog',
        'LVEOG': 'eog',
        'LN': 'misc',  
        'ANG': 'misc', 
        'SP': 'misc',  
        'SD': 'misc',
        'LMAST': 'misc',
        'RMAST': 'misc',
    })
    
    # Drop all non-eeg/eog channels (misc, mastoid, etc.)
    raw.pick_types(eeg=True, eog=True)
    # Resample and apply montage
    raw.resample(eeg_settings["SAMPLING_RATE"])
    raw.set_montage(eeg_settings["MONTAGE"])
    raw.info['bads'] = []
    raw.del_proj()

    return raw

def ransac_detect_bad_channels(raw, dataset, eeg_settings, subject, session=None, task=None, run=None, verbose=True, save_plot=True, show_plot=True):
    raw_hp = raw.copy().filter(l_freq=eeg_settings['LOW_CUTOFF_HZ'], h_freq=None)

    epochs_hp = mne.make_fixed_length_epochs(raw_hp, 
                                            duration=eeg_settings["SYNTHETIC_LENGTH"], 
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

    path_plots = os.path.join(config.PLOTS_PATH, dataset.f_name, 'RANSAC_autodetect')
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


def fix_bad_channels(raw, dataset, subject, session=None, task=None, run=None, verbose=True):
    """
    Fix bad channels in the raw data. This function loads the bad channels from a JSON file and updates the raw data accordingly.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw data object to fix bad channels.
    dataset : DatasetConfig
        The dataset configuration object.
    subject : str
        The subject identifier.
    session : str, optional
        The session identifier (default is None).
    task : str, optional
        The task identifier (default is None).
    run : str, optional
        The run identifier (default is None).
    verbose : bool, optional
        If True, print additional information (default is True).
    
    Returns
    -----------
    raw : mne.io.Raw
        The raw data object with updated bad channels.
    """
    path = dataset.path_derivatives

    # Load the bad channels from the JSON file
    bad_channels = load_bad_channels(path, dataset.f_name, subject, session=session, task=task, run=run)
    
    if bad_channels is None or len(bad_channels) == 0:
        if verbose:
            print(f"No bad channels found for subject {subject} with session={session}, task={task}, run={run}. Skipping interpolation.")
        return raw

    # Update the raw data with the bad channels
    raw.info['bads'] = bad_channels

    # Interpolate the bad channels
    raw.interpolate_bads(reset_bads=True, verbose=verbose)
    
    return raw

def autoreject_raw(raw, eeg_settings, verbose=True):
    """
    Apply autoreject to the raw data. This function fits an AutoReject object to the raw data and saves the autoreject object.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw data object to apply autoreject.
    eeg_settings : dict
        The EEG settings dictionary defined in config containing parameters
    verbose : bool, optional
        If True, print additional information (default is True).
    
    Returns
    -----------
    ar : AutoReject
        The fitted AutoReject object.
    """
    epochs = mne.make_fixed_length_epochs(
        raw.copy().filter(l_freq=eeg_settings['LOW_CUTOFF_HZ'], h_freq=None), 
        duration=eeg_settings["SYNTHETIC_LENGTH"], 
        preload=True)
    
    picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True,
                        stim=False, eog=False,
                        include=[], exclude=[])
    
    epochs = epochs.copy().pick(picks_eeg)

    ar = AutoReject(
        n_interpolate=np.array([12]),              # Try only one interpolation as it always uses the highest n_interpolate
        consensus=np.linspace(0.3, 0.7, 7),        # Require some agreement, not too harsh
        thresh_method='bayesian_optimization',     # Use default method
        cv=10,                                     # cross validation: K-fold
        picks=picks_eeg,                           # Only use EEG channels
        n_jobs=cpu_count(),                        # Use all available CPU cores
        random_state=42069,                        # For reproducibility
        verbose=verbose
    )

    # Get the number of epochs to use for training
    train_len = min(eeg_settings["AR_MAX_TRAINING"], len(epochs))
    if verbose:
        print(f"[AUTOREJECT RAW] Training autoreject on {train_len} epochs.")

    ar.fit(epochs[:train_len])

    return ar

def get_bad_epochs_mask(epochs, channel_thresholds):
    """
    Identify bad epochs based on per-channel peak-to-peak thresholds.

    Parameters
    ----------
    epochs : mne.Epochs
        The epoched EEG data.
    channel_thresholds : dict
        Dictionary mapping channel names to peak-to-peak thresholds (in Volts).

    Returns
    -------
    bad_epochs_mask : np.ndarray of bool
        Boolean array where True indicates a bad epoch.
    """
    # Extract EEG data: shape = (n_epochs, n_channels, n_times)
    data = epochs.get_data()

    # Compute peak-to-peak amplitude for each channel in each epoch
    ptp = data.ptp(axis=2)  # shape = (n_epochs, n_channels)

    # Align threshold values with channel order in epochs
    thresh_array = np.array([channel_thresholds[ch] for ch in epochs.ch_names])

    # Create boolean mask: True if any channel in an epoch exceeds its threshold
    bad_epochs_mask = (ptp > thresh_array).any(axis=1)

    return bad_epochs_mask


def plot_bad_epochs_mask(epochs, bad_epochs_mask, orientation='vertical', show_names='auto',
                         aspect='equal', show=True, ax=None):
    """
    Plot a visualization of epochs marked as bad using a boolean mask.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object (used for channel info).
    bad_epochs_mask : np.ndarray of bool
        Boolean array where True marks a bad epoch.
    orientation : 'vertical' | 'horizontal'
        Plot orientation. Default is 'vertical'.
    show_names : 'auto' | int
        Controls how many channel names are shown on axis.
    aspect : str
        Aspect ratio passed to `imshow()`.
    show : bool
        Whether to display the figure.
    ax : matplotlib.axes.Axes | None
        Optional axes object to plot into.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    """
    n_epochs, n_channels = len(epochs), len(epochs.ch_names)
    ch_names = epochs.ch_names

    # Create image: 0 = good, 1 = bad
    image = np.zeros((n_epochs, n_channels))
    image[bad_epochs_mask, :] = 1  # mark bad epochs as 1

    if show_names == 'auto':
        show_names = 1 if n_channels < 25 else 5
    ch_names_shown = ch_names[::show_names]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    ax.grid(False)
    cmap = ListedColormap(['lightgreen', 'red'])  # good, bad
    label_map = {0: 'good', 1: 'bad'}

    if orientation == 'horizontal':
        img = ax.imshow(image.T, cmap=cmap, vmin=0, vmax=1, interpolation='nearest', aspect=aspect)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Channels')
        plt.setp(ax, yticks=range(0, n_channels, show_names), yticklabels=ch_names_shown)
        plt.setp(ax.get_yticklabels(), fontsize=8)

        for idx in np.where(bad_epochs_mask)[0]:
            ax.add_patch(patches.Rectangle((idx - 0.5, -0.5), 1, n_channels,
                                           linewidth=1, edgecolor='r', facecolor='none'))
    elif orientation == 'vertical':
        img = ax.imshow(image, cmap=cmap, vmin=0, vmax=1, interpolation='nearest', aspect=aspect)
        ax.set_xlabel('Channels')
        ax.set_ylabel('Epochs')
        plt.setp(ax, xticks=range(0, n_channels, show_names), xticklabels=ch_names_shown)
        plt.setp(ax.get_xticklabels(), fontsize=8, rotation='vertical')

        for idx in np.where(bad_epochs_mask)[0]:
            ax.add_patch(patches.Rectangle((-0.5, idx - 0.5), n_channels, 1,
                                           linewidth=1, edgecolor='r', facecolor='none'))
    else:
        raise ValueError(f"Invalid orientation: {orientation}")

    # Add legend
    handles = [patches.Patch(color=img.cmap(img.norm(i)), label=label)
               for i, label in label_map.items()]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.tick_params(axis='both', which='both', length=0)
    fig.tight_layout()

    if show:
        plt.show()
    return fig



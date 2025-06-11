import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import pandas as pd
import os
from joblib import cpu_count
from autoreject import AutoReject, Ransac
from mne.preprocessing import ICA
from mne.viz import plot_epochs as plot_mne_epochs
from datetime import datetime

import utils.config as config
from utils.config import DATASETS, set_plot_style
from utils.dataset_config import DatasetConfig
from utils.file_io import load_bad_channels, update_bad_channels_json, log_dropped_epochs, log_reject_threshold, load_new_events
from utils.helpers import plot_ransac_bad_log, encode_events, get_scaled_rejection_threshold


def prepare_raw_data(raw: mne.io.Raw, dataset: DatasetConfig, eeg_settings: dict) -> mne.io.Raw:
    """
    Prepare data for preprocessing. setting correct channel types, selecting channels, 
    resample, set montage, and more deping on the dataset. Function works in place of 
    the raw data object.
    
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
    


def _prepare_braboszcz2017(raw: mne.io.Raw, dataset, eeg_settings):
    # Define desired channel type mappings
    desired_types = {
        'EXG1': 'misc', 'EXG2': 'misc', 'EXG3': 'eog', 'EXG4': 'eog',
        'EXG5': 'eog', 'EXG6': 'eog', 'EXG7': 'misc', 'EXG8': 'misc',
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

    # pick only EEG channels
    raw.pick(['eeg','eog'])

    # Downsample
    raw.resample(eeg_settings["SAMPLING_RATE"])

    # Apply EEG montage
    raw.set_montage(eeg_settings["MONTAGE"])

    # Clear projections and bad channels
    raw.info['bads'] = []
    raw.del_proj()

    return raw

def _prepare_jin2019(raw: mne.io.Raw, dataset, eeg_settings):
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
        'UVEOG': 'eog',
        'LVEOG': 'eog',
        'LHEOG': 'eog',
        'RHEOG': 'eog',
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

def _prepare_touryan2022(raw: mne.io.Raw, dataset, eeg_settings):
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
    raw.pick(['eeg', 'eog'])
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
        random_state=42,   # For reproducibility
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


def fix_bad_channels(raw: mne.io.Raw, dataset: DatasetConfig, subject, session=None, task=None, run=None, verbose=True):
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
    bad_channels = load_bad_channels(path, dataset.f_name, subject, session=session, task=task, run=run, mode='inspect')
    
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
        raw.copy().filter(l_freq=eeg_settings['LOW_CUTOFF_HZ'], h_freq=None, verbose=verbose, n_jobs=cpu_count()), 
        duration=eeg_settings["SYNTHETIC_LENGTH"], 
        preload=True,
        verbose=verbose)
    
    picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True,
                        stim=False, eog=False,
                        include=[], exclude=[])
    
    epochs = epochs.copy().pick(picks_eeg, verbose=verbose)

    ar = AutoReject(
        n_interpolate=np.array([4]),               # Try only one interpolation as it always uses the highest n_interpolate
        consensus=np.linspace(0.7, 0.9, 7),        # Require some agreement, not too harsh
        thresh_method='bayesian_optimization',     # Use default method
        cv=10,                                     # cross validation: K-fold
        picks=picks_eeg,                           # Only use EEG channels
        n_jobs=cpu_count(),                        # Use all available CPU cores
        random_state=42,                           # For reproducibility
        verbose=verbose
    )

    # Get the number of epochs to use for training
    train_len = min(eeg_settings["AR_MAX_TRAINING"], len(epochs))
    if verbose:
        print(f"[AUTOREJECT RAW] Training autoreject on {train_len} epochs.")

    ar.fit(epochs[:train_len])

    return ar

def get_bad_epochs_mask(epochs, channel_thresholds, min_bad_channels=3):
    """
    Identify bad epochs based on how many channels exceed their peak-to-peak threshold.

    Parameters
    ----------
    epochs : mne.Epochs
        The epoched EEG data.
    channel_thresholds : dict
        Dictionary mapping channel names to peak-to-peak thresholds (in Volts).
    min_bad_channels : int
        Minimum number of channels that must exceed their threshold in an epoch
        for that epoch to be marked as bad.

    Returns
    -------
    bad_epochs_mask : np.ndarray of bool
        Boolean array where True indicates a bad epoch.
    """
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    ptp = data.ptp(axis=2)  # peak-to-peak amplitude per epoch/channel

    # Get thresholds in the same channel order
    thresh_array = np.array([channel_thresholds[ch] for ch in epochs.ch_names])

    # Count how many channels exceed their threshold in each epoch
    bad_channel_counts = (ptp > thresh_array).sum(axis=1)  # shape: (n_epochs,)

    # Mark epochs as bad if bad channels ≥ min_bad_channels
    bad_epochs_mask = bad_channel_counts >= min_bad_channels

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

def prepare_ica_epochs(raw, dataset, eeg_settings, subject, session=None, task=None, run=None, min_threshold=300e-6, reject_scale_factor=2, verbose=True):
    """
    Prepare epochs for ICA. This function takes raw, untouched data and creates epochs for ICA.
    The function:
     - downsampels the data
     - sets the montage
     - sets the channel types
     - sets the channel names
     - sets the bad channels based on premade JSON containing the bad channels
     - interpolates bads
     - filters the data
     - creates synthetic epochs
     - removes the bad epochs based on threshold
     - fits ICA to the epochs
     - returns the epochs object
    Parameters
    ----------
    raw : mne.io.Raw
        The raw data object to prepare for ICA.
    dataset : DatasetConfig
        The dataset configuration object.
    eeg_settings : dict
        The EEG settings dictionary defined in config containing parameters
    subject : str
        The subject identifier.
    session : str, optional
        The session identifier (default is None).
    task : str, optional
        The task identifier (default is None).
    run : str, optional
        The run identifier (default is None).
    min_threshold : float, optional
        The minimum threshold for rejecting epochs (default is 300e-6).
    reject_scale_factor : float, optional
        The scale factor to increase the rejection threshold (default is 2).
    verbose : bool, optional
        If True, print additional information (default is True).

    Returns
    -----------
    epochs : mne.Epochs
        The prepared epochs object for ICA.
    reject : dict
        The rejection thresholds for the epochs.
    """
    # Prepare raw data
    raw = prepare_raw_data(raw, dataset, eeg_settings)

    # Fix bad channels
    raw = fix_bad_channels(raw, dataset, subject=subject, session=session, task=task, run=run, verbose=verbose)

    # Filter the raw data
    raw.filter(l_freq=eeg_settings['LOW_CUTOFF_HZ'],
                h_freq=None, 
                n_jobs=cpu_count(), 
                picks=['eeg', 'eog'],
                verbose=verbose)

    # Create synthetic epochs
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=eeg_settings["SYNTHETIC_LENGTH"],
        preload=True,
        verbose=verbose
    )
    epochs.pick(picks=['eeg', 'eog'], verbose=verbose)

    # Average reference
    epochs.set_eeg_reference(ref_channels='average', ch_type='eeg', projection=False, verbose=verbose)

    reject = get_scaled_rejection_threshold(
        epochs=epochs,
        reject_scale_factor=reject_scale_factor,
    )
    # If threhold is too low, set it to min_threshold
    # This is implemented to avoid dropping epochs with ocular artifacts
    if reject['eeg'] < min_threshold:
        reject = {key: min_threshold for key in reject.keys()}

    if verbose:
        print(f"[REJECT THRESHOLD] {reject}")

    # Log the rejection threshold
    log_reject_threshold(
        reject=reject,
        dataset=dataset,
        subject=subject,
        session=session,
        task=task,
        run=run,
        verbose=verbose
    )

    # Remove bad epochs based on rejection threshold
    epochs.drop_bad(reject, verbose=verbose)

    log_dropped_epochs(
        epochs=epochs,
        dataset=dataset,
        subject=subject,
        log_root=config.PREPROCESSING_LOG_PATH,
        stage='pre_ica',
        session=session,
        task=task,
        run=run,
        threshold=reject['eeg'],
        verbose=verbose
    )

    return epochs

def ica_fit(epochs, eeg_settings, random_state=42, verbose=True):
    """
    Fit ICA to the epochs. This function applies ICA to the epochs and returns the fitted ICA object.
    
    Parameters
    ----------
    raw : mne.io.Epochs
        The epochs data object to fit ICA.
    eeg_settings : dict
        The EEG settings dictionary defined in config containing parameters
    n_components : int | None
        Number of components to use for ICA. If None, it will be set to 0.95 of the number of channels.
    random_state : int
        Random state for reproducibility.
    verbose : bool, optional
        If True, print additional information (default is True).
    
    Returns
    -----------
    ica : mne.preprocessing.ICA
        The fitted ICA object.
    """
        
    # Apply ICA to the raw data
    ica = ICA(n_components=eeg_settings["N_ICA_COMPONENTS"], 
              random_state=random_state, 
              max_iter=600,
              method='infomax',
              verbose=verbose,
    )



    ica.fit(epochs)
    
    return ica

# def plot_bad_epochs_only(epochs, bad_epochs_mask, scalings=None, title='Bad Epochs Only'):
#     """
#     Plot only the epochs marked as bad using a boolean mask.

#     Parameters
#     ----------
#     epochs : mne.Epochs
#         The full epochs object.
#     bad_epochs_mask : np.ndarray of bool
#         Boolean array where True indicates a bad epoch.
#     scalings : dict | None
#         Scaling factors for the traces.
#     title : str
#         Title for the plot.

#     Returns
#     -------
#     fig : matplotlib.figure.Figure
#         The matplotlib figure with plotted traces.
#     """
#     if len(bad_epochs_mask) != len(epochs):
#         raise ValueError("bad_epochs_mask must be the same length as the number of epochs.")

#     bad_epoch_indices = np.where(bad_epochs_mask)[0]

#     if len(bad_epoch_indices) == 0:
#         print("No bad epochs to plot.")
#         return None

#     # Select only bad epochs
#     bad_epochs = epochs[bad_epoch_indices]

#     # Create red colors for all traces in each bad epoch
#     n_channels = len(bad_epochs.ch_names)
#     epoch_colors = [['r'] * n_channels for _ in range(len(bad_epochs))]

#     # Plot using MNE's internal plot_epochs
#     fig = plot_mne_epochs(
#         epochs=bad_epochs,
#         events=bad_epochs.events,
#         epoch_colors=epoch_colors,
#         scalings=scalings,
#         title=title,
#         block=True
#     )

#     return fig


def prepare_custom_events(raw, event_id_map):
    """
    Extract events from raw.annotations using a predefined event_id map.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG object containing annotations.
    event_id_map : dict
        A dictionary mapping event code as a string (e.g., "2811") to a human-readable description.

    Returns
    -------
    new_events : np.ndarray
        Events array with the structure (n_events, 3), using desired event IDs.
    final_event_id : dict
        Dictionary mapping human-readable description to the desired event code (as int).
    """

    # Extract raw annotations and convert to events
    events, auto_event_id = mne.events_from_annotations(raw)

    # Reverse the auto-generated event_id to get the original annotation strings
    inv_auto_event_id = {v: k for k, v in auto_event_id.items()}  # E.g., 1 -> 2811

    # Build new event list with our desired event IDs and descriptions
    new_events = []
    final_event_id = {}  # human-readable description -> int code

    for ev in events:
        onset_sample, _, auto_code = ev
        ann_str_code = inv_auto_event_id[auto_code]  # e.g., '2811'

        if ann_str_code in event_id_map:
            description = event_id_map[ann_str_code]
            event_code = int(ann_str_code)
            new_events.append([onset_sample, 0, event_code])
            final_event_id[description] = event_code

    new_events = np.array(new_events)
    return new_events, final_event_id


def add_missed_detection_events(events, missed_code=9999, button_press_codes=[4621, 4611]):
    """
    Identify missed police car detections (2812) where no valid button press occurred after 2811.

    Parameters
    ----------
    events : np.ndarray
        Full MNE-style events array.
    missed_code : int
        Code to assign to missed detections.
    button_press_codes : list of int
        Event codes that count as valid button presses.

    Returns
    -------
    missed_events : np.ndarray
        Events array of missed detections.
    """
    missed_events = []
    # Sort by sample to ensure order
    events = events[np.argsort(events[:, 0])]

    i = 0
    while i < len(events):
        onset_sample, _, code = events[i]
        if code == 2811:
            # Look for matching 2812
            j = i + 1
            while j < len(events) and events[j][2] != 2812:
                j += 1

            if j < len(events):
                end_sample = events[j][0]
                # Check if any valid button press happened in this interval
                press_detected = any(
                    (onset_sample < e[0] < end_sample and e[2] in button_press_codes)
                    for e in events[i+1:j]
                )
                if not press_detected:
                    missed_events.append([end_sample + 1, 0, missed_code])
                i = j + 1  # Move past 2812
            else:
                i += 1  # No end found; skip
        else:
            i += 1

    return np.array(missed_events) if missed_events else np.empty((0, 3), dtype=int)

def fixed_length_epochs_braboszcz(raw: mne.io.BaseRaw, event_id_map, subject_group, task, duration, verbose=True):
    """
    Create fixed-length MNE epochs from continuous data using metadata from a DatasetConfig-like object.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed continuous EEG data (e.g., ICA-cleaned).
    dataset : object
        Dataset configuration object with at least:
            - dataset.subject_group (e.g., "meditators")
            - dataset.task (e.g., "med2")
            - dataset.event_id (e.g., dict mapping task labels to event codes)
    eeg_settings : dict
        EEG settings, should include:
            - "epoch_length": float (in seconds)

    Returns
    -------
    epochs : mne.Epochs
        Fixed-length, linearly detrended epochs object.
    """
    sfreq = raw.info['sfreq']
    n_samples = raw.n_times

    # Calculate sample indices for fixed-length events
    epoch_sample_step = int(duration * sfreq)
    epochs_samples = np.arange(0, n_samples - epoch_sample_step + 1, epoch_sample_step)

    # Get event code from dataset object
    task_label = f"{subject_group}/{task}"
    event_code = event_id_map[f'{subject_group}/{task}']

    # Build MNE-compatible event array: shape (n_events, 3)
    events = np.column_stack([
        epochs_samples,
        np.zeros(len(epochs_samples), dtype=int),
        event_code * np.ones(len(epochs_samples), dtype=int)
    ]).astype(int)

    # Define event_id dict for labeling
    event_id = {task_label: event_code}

    # Create the MNE Epochs object with linear detrending
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0,
        tmax=duration,
        baseline=None,
        detrend=1,  # Linear detrending
        preload=True,
        verbose=verbose,
    )

    return epochs

def create_analysis_epochs(raw, dataset, eeg_settings, subject, item=None, verbose=True):
    """
    Create analysis epochs from the raw data. This function creates epochs for analysis and returns the epochs object.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw data object to create analysis epochs.
    dataset : DatasetConfig
        The dataset configuration object.
    eeg_settings : dict
        The EEG settings dictionary defined in config containing parameters
    subject : str
        The subject identifier.
    item : str, optional
        The item identifier (i.e., session, task or run).
    verbose : bool, optional
        If True, print additional information (default is True).
    
    Returns
    -----------
    epochs : mne.Epochs
        The created analysis epochs object.
    """

    if dataset.f_name == 'braboszcz2017':
        return _create_analysis_epochs_braboszcz2017(raw, dataset, eeg_settings, subject, task=item, verbose=verbose)
    elif dataset.f_name == 'jin2019':
        return _create_analysis_epochs_jin2019(raw, dataset, eeg_settings, subject, session=item, verbose=verbose)
    elif dataset.f_name == 'touryan2022':
        return _create_analysis_epochs_touryan2022(raw, dataset, eeg_settings, subject, run=item, verbose=verbose)
    else:
        raise ValueError(f"Unknown dataset: {dataset.f_name}")
    
def _create_analysis_epochs_braboszcz2017(raw, dataset, eeg_settings, subject, task=None, verbose=True):
    # Subject group
    subject_group = dataset.extra_info["subject_groups"][subject]

    # Classify the state with task
    if task=="med1" or task=="med2":
        state="OT"
    elif task=="think1" or task=="think2":
        state="MW"
    
    # creating fixed-length epochs
    epochs = fixed_length_epochs_braboszcz(
        raw,
        duration=eeg_settings["EPOCH_LENGTH_SEC"],
        event_id_map=dataset.event_id_map,
        subject_group=subject_group,
        task=task,
        verbose=verbose
    )
    # Return dict with epochs object and description key
    return {f"{task}/{state}": epochs}

    
def _create_analysis_epochs_jin2019(raw, dataset, eeg_settings, subject, session=None, verbose=True):
    # Create events with original stim channel
    events_old = mne.find_events(raw, stim_channel='Status', verbose=False)

    # Load CSV-encoded event labels
    new_events, event_info = load_new_events(os.path.join(dataset.path_derivatives, 'events'), subject, session)
    encoded_new_events = encode_events(new_events)

    try:
        # Keep only events in the 10–21 range
        filtered_events = events_old[(events_old[:, 2] >= 10) & (events_old[:, 2] <= 21)]
        assert len(filtered_events) == len(encoded_new_events), "ERROR: Mismatch between filtered events and CSV rows"
    except AssertionError as e:
        print(subject, session, str(e))

    # Replace event codes
    filtered_events[:, 2] = encoded_new_events

    # Create synthetic STIM channel and inject events
    info = mne.create_info(['STIM'], sfreq=raw.info['sfreq'], ch_types=['stim'])
    stim_data = np.zeros((1, raw.n_times))
    for event in filtered_events:
        stim_data[0, event[0]] = event[2]

    raw_stim = mne.io.RawArray(stim_data, info)
    raw.add_channels([raw_stim], force_update_info=True)
    raw.drop_channels(['Status'])

    # Get new events
    events = mne.find_events(raw, stim_channel='STIM', verbose=False)

    # Filter out unused event codes
    unused_event_ids = set(dataset.event_id_map.values()) - set(events[:, 2])
    event_id = {k: v for k, v in dataset.event_id_map.items() if v not in unused_event_ids}

    if len(filtered_events) != len(events):
        print(subject, session, f"Number of events changed from {len(filtered_events)} to {len(events)}")

    # Map events to class labels
    class_event_id = {label: idx + 1 for idx, label in enumerate(dataset.event_classes.keys())}
    classified_events = events.copy()
    for class_label, codes in dataset.event_classes.items():
        for code in codes:
            classified_events[classified_events[:, 2] == code, 2] = class_event_id[class_label]

    # Build per-class Epochs objects
    epochs_dict = {}
    for class_label, class_code in class_event_id.items():
        matching_events = classified_events[classified_events[:, 2] == class_code]
        if len(matching_events) == 0:
            if verbose:
                print(f"[INFO] No events found for class {class_label}, skipping.")
            continue

        class_epochs = mne.Epochs(
            raw,
            matching_events,
            event_id={class_label: class_code},
            tmin=eeg_settings["EPOCH_START_SEC"],
            tmax=eeg_settings["EPOCH_START_SEC"] + eeg_settings["EPOCH_LENGTH_SEC"],
            baseline=None,
            detrend=1,
            reject=None,
            preload=True,
            verbose=verbose
        )
        class_epochs.apply_baseline((None, None))
        epochs_dict[class_label] = class_epochs

    return epochs_dict



def _create_analysis_epochs_touryan2022(raw, dataset, eeg_settings, subject, run=None, verbose=True):
    # Create events from annotations
    events, event_id = prepare_custom_events(raw, dataset.event_id_map)

    # Add missed detection events
    events = np.vstack([events, add_missed_detection_events(events)])

    # Add synthetic post-collision events for uniform epoching
    POST_COLLISION_CODE = 8888
    sfreq = raw.info["sfreq"]
    post_events = []
    for ev in events:
        if ev[2] == 4421:
            shifted_sample = int(ev[0] + sfreq * eeg_settings["EPOCH_LENGTH_SEC"])
            post_events.append([shifted_sample, 0, POST_COLLISION_CODE])

    if post_events:
        events = np.vstack([events, np.array(post_events)])

    # Define uniform window: always tmin=0, tmax=5s
    tmin = 0
    tmax = eeg_settings["EPOCH_LENGTH_SEC"]

    condition_specs = {
        'police_detection/MW': 9999,
        'collision/MW': 4421,
        'police_detection/OT': 4621,
        'collision/OT': POST_COLLISION_CODE,
    }

    epochs_dict = {}
    for label, code in condition_specs.items():
        if code not in events[:, 2]:
            if verbose:
                print(f"[INFO] No events found for '{label}' (code {code}). Skipping.")
            continue
        # remove events that are not in the event_id_map
        events_to_epoch = events[events[:, 2] == code]

        try:
            if verbose:
                print(f"[INFO] Creating epochs for '{label}' (code {code})...")
            epochs = mne.Epochs(
                raw,
                events_to_epoch,
                event_id={label: code},
                tmin=tmin,
                tmax=tmax,
                detrend=1,
                reject_by_annotation=False,
                baseline=None,
                verbose=False,
                preload=True
            )
            epochs_dict[label] = epochs
            if verbose:
                print(f"[INFO] Created epochs for '{label}' with {len(epochs)} epochs.")
        except Exception as e:
            print(f"[WARNING] Failed to create epochs for '{label}': {e}")

    if not epochs_dict:
        raise RuntimeError("No valid epochs could be created for this subject.")

    return epochs_dict


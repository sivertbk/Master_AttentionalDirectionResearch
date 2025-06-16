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
        min_channels=0.0625,  # Using less channels than defualt (4 out of 64 channels)
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
    Creates analysis-ready epochs from preprocessed raw data based on the dataset type.

    This function acts as a dispatcher, calling dataset-specific epoching functions.
    The 'item' parameter specifies a session, task, or run, depending on the dataset's structure.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The preprocessed (e.g., ICA-cleaned) raw data object.
    dataset : DatasetConfig
        The configuration object for the specific dataset.
    eeg_settings : dict
        A dictionary containing EEG processing parameters (e.g., epoch length).
    subject : str
        The identifier for the subject.
    item : str, optional
        An identifier for a specific data segment (e.g., session, task, or run).
        The interpretation of 'item' depends on the dataset. Defaults to None.
    verbose : bool, optional
        If True, print detailed information during processing. Defaults to True.
    
    Returns
    -------
    dict
        A dictionary where keys are condition labels (e.g., "task/state" or "class_label")
        and values are mne.Epochs objects for that condition.
        The exact structure depends on the dataset-specific epoching function.

    Raises
    ------
    ValueError
        If the dataset name specified in `dataset.f_name` is not recognized.
    """

    if dataset.f_name == 'braboszcz2017':
        # For Braboszcz2017, 'item' corresponds to a task.
        return _create_analysis_epochs_braboszcz2017(raw, dataset, eeg_settings, subject, task=item, verbose=verbose)
    elif dataset.f_name == 'jin2019':
        # For Jin2019, 'item' corresponds to a session.
        return _create_analysis_epochs_jin2019(raw, dataset, eeg_settings, subject, session=item, verbose=verbose)
    elif dataset.f_name == 'touryan2022':
        # For Touryan2022, 'item' corresponds to a run.
        return _create_analysis_epochs_touryan2022(raw, dataset, eeg_settings, subject, run=item, verbose=verbose)
    else:
        raise ValueError(f"Unknown dataset: {dataset.f_name}")
    
def _create_analysis_epochs_braboszcz2017(raw, dataset, eeg_settings, subject, task=None, verbose=True):
    """
    Creates fixed-length analysis epochs for the Braboszcz2017 dataset.

    Epochs are created based on the specified task, which determines the mental state (Meditative State/OT or Mind-Wandering/MW).
    The function uses `fixed_length_epochs_braboszcz` to segment the data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The preprocessed raw data object.
    dataset : DatasetConfig
        The configuration object for the Braboszcz2017 dataset.
    eeg_settings : dict
        EEG processing parameters, including "EPOCH_LENGTH_SEC".
    subject : str
        The subject identifier.
    task : str, optional
        The specific task (e.g., "med1", "think1") for which to create epochs. Defaults to None.
    verbose : bool, optional
        If True, print detailed information. Defaults to True.
    
    Returns
    -------
    dict
        A dictionary with a single key representing the task and state (e.g., "med1/OT")
        and the value being the corresponding mne.Epochs object.
    """
    # Determine subject group (e.g., meditators, controls) from dataset configuration.
    subject_group = dataset.extra_info["subject_groups"][subject]

    # Classify the mental state (OT: On-Task, MW: Mind-Wandering) based on the task.
    if task=="med1" or task=="med2":
        state="OT" # Meditative tasks are considered On-Task.
    elif task=="think1" or task=="think2":
        state="MW" # Thinking tasks are considered Mind-Wandering.
    
    # Create fixed-length epochs using a helper function.
    epochs = fixed_length_epochs_braboszcz(
        raw,
        duration=eeg_settings["EPOCH_LENGTH_SEC"],
        event_id_map=dataset.event_id_map, # Mapping from task labels to event codes.
        subject_group=subject_group,
        task=task,
        verbose=verbose
    )
    # Return a dictionary where the key combines task and state, and the value is the Epochs object.
    return {f"{task}/{state}": epochs}

    
def _create_analysis_epochs_jin2019(raw, dataset, eeg_settings, subject, session=None, verbose=True):
    """
    Creates analysis epochs for the Jin2019 dataset based on event markers.

    This function processes events from the original 'Status' channel,
    integrates them with new event labels loaded from a CSV file,
    creates a synthetic 'STIM' channel, and then epochs the data based on classified event types.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The preprocessed raw data object.
    dataset : DatasetConfig
        The configuration object for the Jin2019 dataset.
    eeg_settings : dict
        EEG processing parameters, including "EPOCH_START_SEC" and "EPOCH_LENGTH_SEC".
    subject : str
        The subject identifier.
    session : str, optional
        The specific session for which to create epochs. Defaults to None.
    verbose : bool, optional
        If True, print detailed information. Defaults to True.
    
    Returns
    -------
    dict
        A dictionary where keys are class labels (e.g., "target", "distractor")
        and values are the corresponding mne.Epochs objects.
    """
    # Find events from the original 'Status' stimulation channel.
    events_original_stim_channel = mne.find_events(raw, stim_channel='Status', verbose=False)

    # Load new event labels from a CSV file (presumably manually annotated or corrected).
    # `event_info` might contain additional metadata not used here.
    new_events_from_csv, event_info = load_new_events(os.path.join(dataset.path_derivatives, 'events'), subject, session)
    encoded_new_events_from_csv = encode_events(new_events_from_csv) # Convert string labels to numerical codes.

    try:
        # Filter original events to keep only those in the relevant range (10-21 for this dataset).
        filtered_original_events = events_original_stim_channel[
            (events_original_stim_channel[:, 2] >= 10) & (events_original_stim_channel[:, 2] <= 21)
        ]
        # Ensure the number of filtered original events matches the number of events from the CSV.
        assert len(filtered_original_events) == len(encoded_new_events_from_csv), \
            "ERROR: Mismatch between filtered events from 'Status' channel and events from CSV."
    except AssertionError as e:
        print(f"Subject: {subject}, Session: {session}, {str(e)}")

    # Replace the event codes in the filtered original events with the new encoded event codes from CSV.
    filtered_original_events[:, 2] = encoded_new_events_from_csv

    # Create a new synthetic STIM channel to hold the processed event information.
    stim_channel_info = mne.create_info(['STIM'], sfreq=raw.info['sfreq'], ch_types=['stim'])
    stim_data_array = np.zeros((1, raw.n_times)) # Initialize with zeros.
    for event_sample, _, event_code in filtered_original_events:
        stim_data_array[0, event_sample] = event_code # Place event codes at their respective sample points.

    raw_stim_channel = mne.io.RawArray(stim_data_array, stim_channel_info)
    raw.add_channels([raw_stim_channel], force_update_info=True) # Add the new STIM channel to the raw data.
    raw.drop_channels(['Status']) # Remove the original 'Status' channel.

    # Find events from the newly created 'STIM' channel.
    events_from_new_stim = mne.find_events(raw, stim_channel='STIM', verbose=False)

    # Filter the dataset's event_id map to include only event codes present in the current data.
    # This avoids errors if some defined event types do not occur.
    present_event_codes = set(events_from_new_stim[:, 2])
    active_event_id_map = {
        description: code for description, code in dataset.event_id_map.items() if code in present_event_codes
    }

    if len(filtered_original_events) != len(events_from_new_stim):
        # This check might indicate issues if events were lost or unexpectedly created.
        print(subject, session, f"Number of events changed from {len(filtered_original_events)} to {len(events_from_new_stim)}")

    # Map event codes to broader class labels (e.g., target, non-target) defined in dataset config.
    class_label_to_numeric_id = {label: idx + 1 for idx, label in enumerate(dataset.event_classes.keys())}
    classified_events_array = events_from_new_stim.copy()
    for class_label, specific_event_codes in dataset.event_classes.items():
        for specific_code in specific_event_codes:
            # Replace specific event codes with their general class's numeric ID.
            classified_events_array[classified_events_array[:, 2] == specific_code, 2] = class_label_to_numeric_id[class_label]

    # Create MNE Epochs objects for each class label.
    epochs_per_class_dict = {}
    for class_label, class_numeric_id in class_label_to_numeric_id.items():
        # Select events belonging to the current class.
        events_for_current_class = classified_events_array[classified_events_array[:, 2] == class_numeric_id]
        
        if len(events_for_current_class) == 0:
            if verbose:
                print(f"[INFO] No events found for class '{class_label}', skipping epoch creation for this class.")
            continue

        # Create epochs for the current class.
        class_epochs = mne.Epochs(
            raw,
            events_for_current_class,
            event_id={class_label: class_numeric_id}, # Use class label for clarity.
            tmin=eeg_settings["EPOCH_START_SEC"],
            tmax=eeg_settings["EPOCH_START_SEC"] + eeg_settings["EPOCH_LENGTH_SEC"],
            baseline=None, # Baseline correction will be applied later if needed.
            detrend=1,     # Apply linear detrending.
            reject=None,   # Rejection is typically done before this stage or on the epochs later.
            preload=True,
            verbose=verbose
        )
        class_epochs.apply_baseline((None, None)) # Apply baseline correction across the entire epoch.
        epochs_per_class_dict[class_label] = class_epochs

    return epochs_per_class_dict



def _create_analysis_epochs_touryan2022(raw, dataset, eeg_settings, subject, run=None, verbose=True):
    """
    Creates analysis epochs for the Touryan2022 dataset, handling custom event logic.

    This function extracts events from annotations, adds events for missed detections
    (e.g., a police car appearing but no button press), and creates synthetic post-collision
    events for consistent epoching around collision events. Epochs are then created for
    predefined conditions (combinations of event type and mental state).
    
    Parameters
    ----------
    raw : mne.io.Raw
        The preprocessed raw data object.
    dataset : DatasetConfig
        The configuration object for the Touryan2022 dataset.
    eeg_settings : dict
        EEG processing parameters, including "EPOCH_LENGTH_SEC".
    subject : str
        The subject identifier.
    run : str, optional
        The specific run for which to create epochs. Defaults to None.
    verbose : bool, optional
        If True, print detailed information. Defaults to True.
    
    Returns
    -------
    dict
        A dictionary where keys are condition labels (e.g., "police_detection/MW")
        and values are the corresponding mne.Epochs objects.

    Raises
    ------
    RuntimeError
        If no valid epochs can be created for any of the specified conditions.
    """
    # Prepare custom events from raw annotations using the dataset's event_id map.
    # `event_id_map_from_annotations` maps human-readable descriptions to integer codes.
    events_from_annotations, event_id_map_from_annotations = prepare_custom_events(raw, dataset.event_id_map)

    # Identify and add events for missed police car detections.
    # This involves checking for 'police car appears' (2811) not followed by a button press before 'police car disappears' (2812).
    missed_detection_events_array = add_missed_detection_events(events_from_annotations)
    all_events = np.vstack([events_from_annotations, missed_detection_events_array])


    # Create synthetic "post-collision" events to allow uniform epoching after a collision (event code 4421).
    # These events are placed `EPOCH_LENGTH_SEC` after the actual collision event.
    POST_COLLISION_EVENT_CODE = 8888 # Arbitrary code for synthetic post-collision markers.
    sampling_frequency = raw.info["sfreq"]
    post_collision_event_list = []
    for event_sample, _, event_code in all_events:
        if event_code == 4421: # 4421 is the code for a collision.
            # Calculate the sample point for the synthetic post-collision event.
            shifted_sample_onset = int(event_sample + sampling_frequency * eeg_settings["EPOCH_LENGTH_SEC"])
            post_collision_event_list.append([shifted_sample_onset, 0, POST_COLLISION_EVENT_CODE])

    if post_collision_event_list:
        all_events = np.vstack([all_events, np.array(post_collision_event_list)])

    # Define a uniform window for all epochs: tmin=0, tmax=EPOCH_LENGTH_SEC.
    epoch_tmin = 0
    epoch_tmax = eeg_settings["EPOCH_LENGTH_SEC"]

    # Define conditions of interest: a mapping from a descriptive label to an event code.
    # MW = Mind Wandering, OT = On Task (attentive).
    # 9999: Missed police detection (implies MW)
    # 4421: Actual collision event (implies MW at time of collision)
    # 4621: Successful police detection (button press, implies OT)
    # POST_COLLISION_EVENT_CODE (8888): Synthetic event after collision (used to epoch post-collision period, implies OT if task resumed)
    # Note: The mental state (MW/OT) association here is based on the event type itself.
    condition_specifications = {
        'police_detection/MW': 9999,    # Missed detection, classified as Mind-Wandering
        'collision/MW': 4421,           # Collision event, classified as Mind-Wandering
        'police_detection/OT': 4621,    # Successful detection, classified as On-Task
        'collision/OT': POST_COLLISION_EVENT_CODE, # Post-collision period, potentially On-Task
    }

    epochs_per_condition_dict = {}
    for condition_label, event_code_for_condition in condition_specifications.items():
        # Check if the event code for the current condition exists in the data.
        if event_code_for_condition not in all_events[:, 2]:
            if verbose:
                print(f"[INFO] No events found for condition '{condition_label}' (code {event_code_for_condition}). Skipping epoch creation.")
            continue
        
        # Select events that match the current condition's event code.
        events_for_current_condition = all_events[all_events[:, 2] == event_code_for_condition]

        try:
            if verbose:
                print(f"[INFO] Creating epochs for condition '{condition_label}' (code {event_code_for_condition})...")
            
            # Create epochs for the current condition.
            condition_epochs = mne.Epochs(
                raw,
                events_for_current_condition,
                event_id={condition_label: event_code_for_condition}, # Use the descriptive label.
                tmin=epoch_tmin,
                tmax=epoch_tmax,
                detrend=1,                      # Apply linear detrending.
                reject_by_annotation=False,     # Do not reject based on 'BAD_' annotations here.
                baseline=None,                  # No baseline correction at this stage.
                verbose=False,                  # Reduce MNE's verbosity for this specific call.
                preload=True
            )
            epochs_per_condition_dict[condition_label] = condition_epochs
            if verbose:
                print(f"[INFO] Created epochs for '{condition_label}' with {len(condition_epochs)} epochs.")
        except Exception as e:
            # Catch potential errors during epoch creation for a specific condition.
            print(f"[WARNING] Failed to create epochs for condition '{condition_label}': {e}")

    if not epochs_per_condition_dict:
        # If no epochs were created for any condition, raise an error.
        raise RuntimeError("No valid epochs could be created for this subject and run.")

    return epochs_per_condition_dict


import os
import mne
from autoreject import read_auto_reject, validation_curve, get_rejection_threshold, set_matplotlib_defaults
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from joblib import cpu_count

from utils.preprocessing_tools import prepare_raw_data, fix_bad_channels, get_bad_epochs_mask
from utils.file_io import load_raw_data, load_ar
from utils.config import DATASETS, set_plot_style, EEG_SETTINGS, PLOTS_PATH
from utils.helpers import cleanup_memory, iterate_dataset_items

set_plot_style()

VERBOSE = True
SHOW_PLOTS = False


def plot_autoreject_validation_curve(epochs, subject, dataset_name, label, item, save_dir, manual_threshold=None, show=False):
    """
    Plot the validation curve for global rejection threshold using AutoReject.

    Parameters
    ----------
    epochs : mne.Epochs
        Preprocessed EEG epochs (only EEG channels).
    subject : str or int
        Subject ID.
    dataset_name : str
        Name of the dataset.
    label : str
        Name of the iteration variable (e.g. 'session', 'task', 'run').
    item : str or int
        Value of the iteration variable (e.g. session ID or run number).
    save_dir : str
        Directory where the plot should be saved.
    manual_threshold : float or None
        A manually chosen threshold to plot (in Volts). Optional.
    show : bool
        Whether to display the plot interactively.
    """
    os.makedirs(save_dir, exist_ok=True)

    set_matplotlib_defaults(plt)

    param_range = np.linspace(40e-6, 200e-6, 30)
    _, test_scores, param_range = validation_curve(
        epochs, param_range=param_range, cv=5, return_param_range=True, n_jobs=1)

    test_scores = -test_scores.mean(axis=1)
    best_thresh = param_range[np.argmin(test_scores)]
    bayes_opt_thresh = get_rejection_threshold(epochs, random_state=0, cv=5)['eeg']

    unit = r'$\mu$V'
    scaling = 1e6

    plt.figure(figsize=(8, 5))
    colors = ['#E24A33', '#348ABD', '#988ED5', 'k']

    plt.plot(scaling * param_range, scaling * test_scores,
             'o-', markerfacecolor='w',
             color=colors[0], markeredgewidth=2, linewidth=2,
             markeredgecolor=colors[0], markersize=8, label='CV scores')

    plt.ylabel(f'RMSE ({unit})')
    plt.xlabel(f'Threshold ({unit})')
    plt.xlim((scaling * param_range[0] * 0.9, scaling * param_range[-1] * 1.1))

    plt.axvline(scaling * best_thresh, label='auto global', color=colors[2], linewidth=2, linestyle='--')
    plt.axvline(scaling * bayes_opt_thresh, label='bayes opt', color=colors[3], linewidth=2, linestyle='--')

    if manual_threshold is not None:
        plt.axvline(scaling * manual_threshold, label='manual', color=colors[1], linewidth=2, linestyle=':')

    plt.legend(loc='upper right')
    title = f"Validation Curve | Dataset: {dataset_name} | Subject: {subject} | {label}: {item}"
    plt.title(title)
    plt.tight_layout()

    filename = f"sub-{subject}_{label}-{item}_ar_validation_curve.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)

    if show:
        plt.show()
    else:
        plt.close()


for dataset, subject, label, item, kwargs in iterate_dataset_items(DATASETS):
    ar = load_ar(dataset, subject, label, item, verbose=VERBOSE)
    if ar is None:
        continue
    
    # Load and prepare raw data
    raw = load_raw_data(dataset, subject, **kwargs, verbose=VERBOSE)
    if raw is None:
        tqdm.write(f"    No data found for subject {subject} with {label} {item}.")
        continue

    raw = prepare_raw_data(raw, dataset, EEG_SETTINGS)    
    
    picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True,
                        stim=False, eog=False,
                        include=[], exclude=[])
    
    raw.pick(picks_eeg)

    # Fix bad channels in the raw data
    raw = fix_bad_channels(raw, dataset, subject, **kwargs, verbose=VERBOSE)

    # Filter the raw data
    raw.filter(l_freq=EEG_SETTINGS['LOW_CUTOFF_HZ'], h_freq=None, n_jobs=cpu_count())

    epochs = mne.make_fixed_length_epochs(
        raw, 
        duration=EEG_SETTINGS["SYNTHETIC_LENGTH"], 
        preload=True)

    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
    tqdm.write(f"[THRESHOLD] {ar.threshes_}")

    # Get the bad epochs mask based on channel thresholds
    bad_epochs_mask = get_bad_epochs_mask(epochs, ar.threshes_)

    # Save the autoreject performance

    path_plots = os.path.join(PLOTS_PATH, dataset.f_name)

    # bad_epoch_inds = reject_log.bad_epochs
    # if bad_epoch_inds is not None and bad_epoch_inds.any():
    #     save_path = os.path.join(path_plots, "bad_epochs_pre_ICA")
    #     os.makedirs(save_path, exist_ok=True)
    #     fig_ar_epochs_before_ica = epochs[bad_epoch_inds].plot(
    #         scalings=dict(eeg=100e-6),
    #         show=SHOW_PLOTS,
    #         block=SHOW_PLOTS
    #     )
    #     fig_ar_epochs_before_ica.suptitle(
    #         f"AutoReject bad epochs pre ICA | Dataset: {dataset.name} | Subject: {subject} | {label}: {item}."
    #     )
    #     fig_ar_epochs_before_ica.savefig(os.path.join(save_path, f"sub-{subject}_{label}-{item}_bad_epochs_pre_ICA.png"))
    # else:
    #     tqdm.write(f"No bad epochs to plot for subject {subject} | {label}: {item}.")

    # plot_bad_epochs_mask(epochs, bad_epochs_mask, orientation='horizontal', show=SHOW_PLOTS)


    # reject_log.plot_epochs(epochs, scalings=dict(eeg=100e-6))

    # save_path = os.path.join(path_plots, "bad_epochs_pre_ICA_matrix")
    # os.makedirs(save_path, exist_ok=True)
    # fig_ar_matrix_before_ica = reject_log.plot(orientation="horizontal", show=SHOW_PLOTS)
    # fig_ar_matrix_before_ica.axes[0].set_title(f"AutoReject bad/interpolated epochs pre ICA | Dataset: {dataset.name} | Subject: {subject} | {label}: {item}.")
    # fig_ar_matrix_before_ica.savefig(os.path.join(save_path , f"sub-{subject}_{label}-{item}_bad_epochs_pre_ICA_matrix.png"))

    # save_path = os.path.join(path_plots, "validation_curves")
    # plot_autoreject_validation_curve(
    #     epochs,
    #     subject=subject,
    #     dataset_name=dataset.name,
    #     label=label,
    #     item=item,
    #     save_dir=save_path,
    #     manual_threshold=140e-6,
    #     show=SHOW_PLOTS
    # )

    # # save reject log to derivatives
    # save_path = os.path.join(dataset.path_derivatives, "pre_ica_autoreject_logs")
    # os.makedirs(save_path, exist_ok=True)
    # reject_log.save(os.path.join(save_path, f"sub-{subject}_{label}-{item}_autoreject_log.npz"), overwrite=True)

import os
import mne
import numpy as np
import warnings
from mne.preprocessing import create_eog_epochs
from mne import set_bipolar_reference
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils.config import DATASETS, set_plot_style, PLOTS_PATH
from utils.helpers import iterate_dataset_items
from utils.file_io import load_ica, save_ica_excluded_components

warnings.filterwarnings("ignore")  # Optional: suppress verbose MNE warnings
set_plot_style()

VERBOSE = True
SHOW_PLOTS = False  # Set False to avoid popups

z_threshold_blink = 5.0
z_threshold_saccade = 5.0

# Datasets
DATASETS.pop('braboszcz2017')
#DATASETS.pop('jin2019', None)

def epochs_to_raw(epochs):
    """Convert MNE Epochs to a Raw object by stitching epochs together."""
    info = epochs.info.copy()

    # Get the epochs data
    data = epochs.get_data()  # shape (n_epochs, n_channels, n_times)

    # Concatenate the epochs along time axis
    data = np.concatenate(data, axis=1)  # now (n_channels, total_timepoints)

    # Create Raw object
    raw = mne.io.RawArray(data, info)

    return raw

def _lighter_red(iteration, max_iterations=3):
    """Return lighter shades of red based on iteration."""
    base_color = np.array([1.0, 0.0, 0.0])  # Pure red (RGB)
    white = np.array([1.0, 1.0, 1.0])  # White
    factor = min(iteration / max_iterations, 1.0)  # Gradually move toward white
    color = base_color + (white - base_color) * factor
    return tuple(color)

def plot_ica_component_scores(
    scores,
    z_threshold,
    title,
    save_dir,
    filename,
    figsize=(12, 4),
    ):
    """
    Plot ICA component scores with iterative adaptive threshold visualization.

    Parameters
    ----------
    scores : np.ndarray
        Correlation scores for ICA components.
    z_threshold : float
        Z-score threshold for exclusion.
    title : str
        Title of the plot.
    save_dir : str
        Directory to save the figure.
    filename : str
        Filename for saving the figure.
    color_cycle : list of str, optional
        List of colors to use for different exclusion rounds.
    figsize : tuple of float, optional
        Size of the figure.
    """

    if scores is None or len(scores) == 0:
        print("No scores to plot.")
        return

    remaining_scores = scores.copy()
    n_components = len(scores)

    fig, ax = plt.subplots(figsize=figsize)

    excluded_components = []
    excluded_colors = []

    iteration = 0
    while True:
        if np.all(np.isnan(remaining_scores)):
            break  # No more scores left

        # Recompute z-scores excluding NaNs
        mean_now = np.nanmean(remaining_scores)
        std_now = np.nanstd(remaining_scores)
        z_scores_iter = (remaining_scores - mean_now) / std_now

        max_z = np.nanmax(np.abs(z_scores_iter))
        if max_z > z_threshold:
            idx = np.nanargmax(np.abs(z_scores_iter))
            excluded_components.append(idx)
            excluded_colors.append(_lighter_red(iteration))

            # Mark as excluded
            remaining_scores[idx] = np.nan

            # Plot threshold lines for this iteration
            thresh_value = z_threshold * std_now
            ax.axhline(y=mean_now + thresh_value, linestyle='--', color=_lighter_red(iteration), alpha=0.6, linewidth=0.8)
            ax.axhline(y=mean_now - thresh_value, linestyle='--', color=_lighter_red(iteration), alpha=0.6, linewidth=0.8)

            iteration += 1
        else:
            break

    # Final bar plot
    bars = ax.bar(np.arange(n_components), scores, color='gray', edgecolor='k')

    for idx, color in zip(excluded_components, excluded_colors):
        bars[idx].set_color(color)
        bars[idx].set_edgecolor('black')

    ax.set_xlabel('ICA components')
    ax.set_ylabel('Score (correlation)')
    ax.set_title(title)
    ax.axhline(y=0, color='black', linewidth=0.8)

    # --- Custom grey proxy artist for legend ---
    proxy_line = Line2D(
        [0], [0],
        color='gray',
        linestyle='--',
        linewidth=1,
        alpha=0.7
    )
    ax.legend([proxy_line], [f'Adaptive threshold: ±{z_threshold} z-score'], loc='upper right')

    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path)
    plt.close(fig)

    print(f"Saved plot to: {save_path}")

def add_title_above_properties(fig, title, height=0.08, fontsize=14):
    """Move all axes down to make space and add a clean title above."""
    for ax in fig.axes:
        pos = ax.get_position()
        new_pos = [pos.x0, pos.y0 - height, pos.width, pos.height]
        ax.set_position(new_pos)
    fig.text(0.5, 0.98, title, ha='center', va='top', fontsize=fontsize)

for dataset, subject, label, item, kwargs in iterate_dataset_items(DATASETS):
    print(f"\n\nProcessing Subject: {subject}, Item: {item}")

    # --- Load data
    file_path = os.path.join(dataset.path_epochs, 'ica_epochs', f"sub-{subject}_{label}-{item}_ica-epo.fif")
    if not os.path.exists(file_path):
        print(f"    No data found for subject {subject} with {label} {item}.")
        continue
    epochs = mne.read_epochs(os.path.join(dataset.path_epochs, 'ica_epochs', f"sub-{subject}_{label}-{item}_ica-epo.fif"), preload=True)
    ica = load_ica(dataset, subject, **kwargs, verbose=VERBOSE)

    # --- Create bipolar EOG channels
    epochs = set_bipolar_reference(epochs, anode='UVEOG', cathode='LVEOG', ch_name='VEOG', drop_refs=True, copy=True)
    epochs = set_bipolar_reference(epochs, anode='LHEOG', cathode='RHEOG', ch_name='HEOG', drop_refs=True, copy=True)

    # --- Find bad EOG components
    blink_inds, blink_scores = ica.find_bads_eog(epochs, ch_name='VEOG', threshold=z_threshold_blink, measure='zscore')
    saccade_inds, saccade_scores = ica.find_bads_eog(epochs, ch_name='HEOG', threshold=z_threshold_saccade, measure='zscore')
    print(f"  Blink components: {blink_inds}, Saccade components: {saccade_inds}")

    # --- Analyze blink score distribution
    plot_ica_component_scores(
        scores=blink_scores,
        z_threshold=z_threshold_blink,
        title=f'{dataset.name} | Sub-{subject} | {label}-{item} | Blink Scores',
        save_dir=os.path.join(PLOTS_PATH, dataset.f_name, 'ica_scores_blinks'),
        filename=f"sub-{subject}_{label}-{item}_blink_scores.png",
    )

    # --- Analyze saccade score distribution
    plot_ica_component_scores(
        scores=saccade_scores,
        z_threshold=z_threshold_saccade,
        title=f'{dataset.name} | Sub-{subject} | {label}-{item} | Saccade Scores',
        save_dir=os.path.join(PLOTS_PATH, dataset.f_name, 'ica_scores_saccades'),
        filename=f"sub-{subject}_{label}-{item}_saccade_scores.png",
    )


    # --- Make a Raw object from epochs
    raw = epochs_to_raw(epochs)

    # --- Create EOG Evoked (before cleaning)
    eog_epochs_before = create_eog_epochs(raw, ch_name='VEOG', verbose=False)
    eog_evoked_before = eog_epochs_before.average()
    eog_evoked_before.apply_baseline((None, -0.2))

    # --- Apply ICA (cleaning)
    ica.exclude = blink_inds + saccade_inds
    raw_clean = raw.copy()
    ica.apply(raw_clean)

    # --- Create EOG Evoked (after cleaning)
    eog_epochs_after = create_eog_epochs(raw_clean, ch_name='VEOG', verbose=False)
    eog_evoked_after = eog_epochs_after.average()
    eog_evoked_after.apply_baseline((None, -0.2))



    # --- Define save path
    save_dir = os.path.join(PLOTS_PATH, dataset.f_name, 'ica_properties')
    os.makedirs(save_dir, exist_ok=True)

    if len(blink_inds) != 0:
        # --- Plot ICA properties for blinks and save
        figs_blink = ica.plot_properties(epochs, picks=blink_inds, psd_args={'fmax': 40.}, show=SHOW_PLOTS)

        if not isinstance(figs_blink, list):
            figs_blink = [figs_blink]

        for i, fig in enumerate(figs_blink):
            comp_idx = blink_inds[i]
            title = f'{dataset.name} | Sub-{subject} | {label}-{item} | Blink Component {comp_idx}'
            add_title_above_properties(fig, title, height=0.06)  # tune height if needed
            fig.savefig(os.path.join(save_dir, f"sub-{subject}_{label}-{item}_blink_comp-{comp_idx:03d}.png"))
            plt.close(fig)



    if len(saccade_inds) != 0:
        # --- Plot ICA properties for saccades and save
        figs_saccade = ica.plot_properties(epochs, picks=saccade_inds, psd_args={'fmax': 40.}, show=SHOW_PLOTS)

        if not isinstance(figs_saccade, list):
            figs_saccade = [figs_saccade]
        for i, fig in enumerate(figs_saccade):
            comp_idx = saccade_inds[i]
            title = f'{dataset.name} | Sub-{subject} | {label}-{item} | Saccade Component {comp_idx}'
            add_title_above_properties(fig, title, height=0.06)
            fig.savefig(os.path.join(save_dir, f"sub-{subject}_{label}-{item}_saccade_comp-{comp_idx:03d}.png"))
            plt.close(fig)


    # --- Plot EOG evoked and save
    fig_before = eog_evoked_before.plot_joint(show=SHOW_PLOTS, title=f'{dataset.name} | Sub-{subject} | {label}-{item} | BEFORE ICA')
    fig_after = eog_evoked_after.plot_joint(show=SHOW_PLOTS, title=f'{dataset.name} | Sub-{subject} | {label}-{item} | AFTER ICA')

    # Output folder
    save_dir = os.path.join(PLOTS_PATH, dataset.f_name, 'eog_evoked_before_after')
    os.makedirs(save_dir, exist_ok=True)

    fig_before.savefig(os.path.join(save_dir, f"sub-{subject}_{label}-{item}_before.png"))
    fig_after.savefig(os.path.join(save_dir, f"sub-{subject}_{label}-{item}_after.png"))
    plt.close('all')  # Close figures to avoid memory leaks





    # --- Plot EOG overlay and save
    fig_overlay = ica.plot_overlay(raw, exclude=blink_inds + saccade_inds, picks='eeg', show=SHOW_PLOTS)
    fig_overlay.suptitle(f'{dataset.name} | Sub-{subject} | {label}-{item} | EOG Overlay')

    save_dir = os.path.join(PLOTS_PATH, dataset.f_name, 'eog_overlay')
    os.makedirs(save_dir, exist_ok=True)

    fig_overlay.savefig(os.path.join(save_dir, f"sub-{subject}_{label}-{item}_eog_overlay.png"))
    plt.close('all')  # Close figures to avoid memory leaks

    # --- Save list of excluded components to derivatives json
    save_ica_excluded_components(
        dataset=dataset,
        subject=subject,
        label=label,
        item=item,
        blink_components=blink_inds,
        saccade_components=saccade_inds
    )

    print(f"  Finished processing Subject: {subject}, Item: {item}")

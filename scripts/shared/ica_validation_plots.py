import os
import mne
import numpy as np
from mne.preprocessing import create_eog_epochs
from mne import set_bipolar_reference
import matplotlib.pyplot as plt
from utils.config import DATASETS, set_plot_style, PLOTS_PATH
from utils.helpers import iterate_dataset_items
from utils.file_io import load_ica
import warnings

warnings.filterwarnings("ignore")  # Optional: suppress verbose MNE warnings
set_plot_style()

VERBOSE = True
SHOW_PLOTS = False  # Set False to avoid popups

# Datasets
DATASETS.pop('braboszcz2017')
# DATASETS.pop('jin2019', None)

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
    blink_inds, blink_scores = ica.find_bads_eog(epochs, ch_name='VEOG', threshold=4.5, measure='zscore')
    saccade_inds, saccade_scores = ica.find_bads_eog(epochs, ch_name='HEOG', threshold=5.0, measure='zscore')
    print(f"  Blink components: {blink_inds}, Saccade components: {saccade_inds}")

    # --- Analyze blink score distribution
    if blink_scores is not None:
        z_scores = (blink_scores - np.mean(blink_scores)) / np.std(blink_scores)
        print(f"  Max blink score: {np.max(blink_scores):.3f}")
        print(f"  Max blink z-score: {np.max(z_scores):.3f}")

        # Show what would be detected at lower threshold (e.g., 3.0)
        lower_thresh = 3.0
        blink_inds_lo_thresh = list(np.where(z_scores > lower_thresh)[0])
        print(f"  Components at z > {lower_thresh}: {blink_inds_lo_thresh}")

        # --- Plot blink scores
        fig, ax = plt.subplots(figsize=(10, 3))
        bars = ax.bar(np.arange(len(blink_scores)), blink_scores, color='gray', edgecolor='black')
        for i in blink_inds:
            bars[i].set_color('red')  # Highlight excluded components
        ax.set_xlabel('ICA components')
        ax.set_ylabel('score')
        ax.set_title(f'{dataset.name} | Sub-{subject} | {label}-{item} | Blink Scores')
        ax.axhline(y=np.mean(blink_scores) + 4.5 * np.std(blink_scores), linestyle='--', color='red', label='threshold')
        ax.legend()
        fig.tight_layout()

        # Save or show
        save_dir = os.path.join(PLOTS_PATH, dataset.f_name, 'ica_scores')
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"sub-{subject}_{label}-{item}_blink_scores.png"))
        plt.close(fig)

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




    # --- Plot scores and save
    fig_blink_scores = ica.plot_scores(blink_scores, show=SHOW_PLOTS)
    fig_blink_scores.suptitle(f'{dataset.name} | Sub-{subject} | {label}-{item} | Blink Scores')

    save_dir = os.path.join(PLOTS_PATH, dataset.f_name, 'blink_scores')
    os.makedirs(save_dir, exist_ok=True)

    fig_blink_scores.savefig(os.path.join(save_dir, f"sub-{subject}_{label}-{item}_blink_scores.png"))
    plt.close('all')  # Close figures to avoid memory leaks




    fig_saccade_scores = ica.plot_scores(saccade_scores, show=SHOW_PLOTS)
    fig_saccade_scores.suptitle(f'{dataset.name} | Sub-{subject} | {label}-{item} | Saccade Scores')

    save_dir = os.path.join(PLOTS_PATH, dataset.f_name, 'saccade_scores')
    os.makedirs(save_dir, exist_ok=True)

    fig_saccade_scores.savefig(os.path.join(save_dir, f"sub-{subject}_{label}-{item}_saccade_scores.png"))
    plt.close('all')  # Close figures to avoid memory leaks




    # --- Plot ICA properties for blinks and save
    figs_blink = ica.plot_properties(epochs, picks=blink_inds, psd_args={'fmax': 40.}, show=SHOW_PLOTS)

    # --- Define save path
    save_dir = os.path.join(PLOTS_PATH, dataset.f_name, 'ica_properties')
    os.makedirs(save_dir, exist_ok=True)

    if not isinstance(figs_blink, list):
        figs_blink = [figs_blink]

    for i, fig in enumerate(figs_blink):
        comp_idx = blink_inds[i]
        title = f'{dataset.name} | Sub-{subject} | {label}-{item} | Blink Component {comp_idx}'
        add_title_above_properties(fig, title, height=0.06)  # tune height if needed
        fig.savefig(os.path.join(save_dir, f"sub-{subject}_{label}-{item}_blink_comp-{comp_idx:03d}.png"))
        plt.close(fig)




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




    # --- Create EOG Evoked (after cleaning)
    eog_epochs_after = create_eog_epochs(raw_clean, ch_name='VEOG', verbose=False)
    eog_evoked_after = eog_epochs_after.average()
    eog_evoked_after.apply_baseline((None, -0.2))

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

    print(f"  Finished processing Subject: {subject}, Item: {item}")

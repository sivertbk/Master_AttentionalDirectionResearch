import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mne.preprocessing import create_eog_epochs
from mne import set_bipolar_reference

from utils.config import DATASETS, set_plot_style, EEG_SETTINGS, PLOTS_PATH
from utils.helpers import iterate_dataset_items, get_scaled_rejection_threshold
from utils.file_io import (load_ica, load_raw_data, load_ica_excluded_components, 
                           log_reject_threshold, log_dropped_epochs, save_epochs_dict)
from utils.preprocessing_tools import prepare_raw_data, fix_bad_channels, create_analysis_epochs
from .dataset_spesific.jin2019.jin2019_probe_extraction import extract_jin2019_probe_data 

set_plot_style()

PLOT_EOG_RAW = False


def print_epoch_counts(epochs):
    counts = {}
    for label, code in epochs.event_id.items():
        n = np.sum(epochs.events[:, 2] == code)
        counts[label] = n
        print(f"{label}: {n} epochs")
    return counts

def get_dropped_epoch_indices(drop_log):
    return [i for i, log in enumerate(drop_log) if len(log) > 0]

def plot_overlapping_psd(epochs_before, epochs_after, fmin=1, fmax=40, picks="eeg", title=None, save_path=None):
    # Compute PSDs
    psd_bef = epochs_before.compute_psd(fmin=fmin, fmax=fmax, picks=picks)
    psd_aft = epochs_after.compute_psd(fmin=fmin, fmax=fmax, picks=picks)

    freqs = psd_bef.freqs

    # Average across epochs and channels
    psd_bef_avg = psd_bef.get_data().mean(axis=0).mean(axis=0)
    psd_aft_avg = psd_aft.get_data().mean(axis=0).mean(axis=0)

    # Convert to dB
    psd_bef_db = 10 * np.log10(psd_bef_avg)
    psd_aft_db = 10 * np.log10(psd_aft_avg)

    # Determine good ylim based on both signals
    psd_all = np.concatenate([psd_bef_db, psd_aft_db])
    lower = np.floor(psd_all.min() - 1)
    upper = np.ceil(psd_all.max() + 1)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, psd_bef_db, color='red', alpha=0.6, label='Before rejection')
    plt.plot(freqs, psd_aft_db, color='black', alpha=0.9, label='After rejection')
    plt.fill_between(freqs, psd_bef_db, psd_aft_db, color='gray', alpha=0.3)

    plt.ylim([lower, upper])
    plt.title(title or "PSD Before vs After Epoch Rejection")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (dB)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved PSD plot to {save_path}")

    plt.clf()
    plt.close()

def max_ptp_evoked(evoked1, evoked2):
    """Return the maximum peak-to-peak value across two Evoked objects."""
    ptp1 = np.ptp(evoked1.data, axis=1).max()  # ptp per channel, then max across channels
    ptp2 = np.ptp(evoked2.data, axis=1).max()
    return max(ptp1, ptp2)


if __name__ == "__main__":
    # Run the probe extraction for the Jin2019 dataset to create necessary CSV files for step 4.
    # This is a one-time operation to prepare the data for further processing.
    extract_jin2019_probe_data()

    for dataset, subject, label, item, kwargs in iterate_dataset_items(DATASETS):
        ##----------------------------------------------------------------------------##
        #                    1. LOAD AND PREPARE RAW EEG DATA                          #
        ##----------------------------------------------------------------------------##
        ''' Load the raw data and create copies for further processing. All channel
            types will be set correctly. Data will be downsampled to sampling rate
            defined in config.py.

            RESULTS: 
            - Raw data objects from the respective dataset ready to be preprocessed.
        '''
        raw = load_raw_data(dataset, subject, **kwargs, verbose=True)

        if raw is None:
            print(f"[WARN] No data found for subject {subject} with {label} {item}.")
            continue

        prepare_raw_data(raw, dataset, EEG_SETTINGS)

        ##----------------------------------------------------------------------------##
        #               2. INTERPOLATE BAD CHANNELS IN ORIGINAL RAW                    #
        ##----------------------------------------------------------------------------##
        ''' Interpolate bad channels in the original raw data based on RANSAC results
            which is also visually inspected and updated and saved as JSONs for each
            dataset, subject, and session/task/run.

            RESULTS:
            - Cleaned raw object with interpolated channels
        '''
        fix_bad_channels(raw, dataset, subject=subject, **kwargs, verbose=True)

        ##----------------------------------------------------------------------------##
        #            3. APPLY ICA TO FULL UNFILTERED RAW (CLEANED VERSION)             #
        ##----------------------------------------------------------------------------##
        ''' Apply the ICA solution (trained on clean synthetic epochs) to the full,
            unfiltered raw data and exluding blink and saccade related artifacts 
            automatic using the provided EOG channels for correlation matching
            with the provided ICA components. ICA models are fitted using the 
            fit_ica.py script, and artifacts are detected using the ica_artifact_detection.py

            Note: The models were trained on average referenced data, thus the raw
            data is also average referenced first.

            RESULTS:
            - ICA-cleaned raw object
        '''
        # Always average reference the raw data before applying ICA (will be done to data not ICA-cleaned as well)
        raw.set_eeg_reference('average', ch_type='eeg', verbose=True, projection=False)

        ica = load_ica(dataset, subject, **kwargs, verbose=True)
        if ica is None:
            print(f"[WARN] No ICA found for subject {subject} with {label} {item}. Skipping ICA application.")
            
        components_to_exclude = load_ica_excluded_components(dataset, subject, label, item)
        if components_to_exclude is None:
            print(f"[WARN] No excluded components found for subject {subject} with {label} {item}.")
        
        else:
            # --- Create bipolar EOG channels
            raw = set_bipolar_reference(raw, anode='UVEOG', cathode='LVEOG', ch_name='VEOG', drop_refs=True, copy=True)
            raw = set_bipolar_reference(raw, anode='LHEOG', cathode='RHEOG', ch_name='HEOG', drop_refs=True, copy=True)

            raw_before = raw.copy()
            
            raw = ica.apply(raw, exclude=components_to_exclude, verbose=True)

            if PLOT_EOG_RAW:
                # --- Create EOG Evoked (before cleaning)
                eog_epochs_before = create_eog_epochs(raw_before, reject=dict(eeg=400e-6), ch_name='VEOG', verbose=False)
                eog_evoked_before = eog_epochs_before.average()
                eog_evoked_before.apply_baseline((None, -0.2))

                # --- Create EOG Evoked (after cleaning)
                eog_epochs_after = create_eog_epochs(raw, reject=dict(eeg=400e-6), ch_name='VEOG', verbose=False)
                eog_evoked_after = eog_epochs_after.average()
                eog_evoked_after.apply_baseline((None, -0.2))
                
                # --- Plot EOG evoked and save
                fig_before = eog_evoked_before.plot_joint(show=False, title=f'{dataset.name} | Sub-{subject} | {label}-{item} | BEFORE ICA')
                fig_after = eog_evoked_after.plot_joint(show=False, title=f'{dataset.name} | Sub-{subject} | {label}-{item} | AFTER ICA')

                # Output folder
                save_dir = os.path.join(PLOTS_PATH, dataset.f_name, 'eog_evoked_before_after_raw_data')
                os.makedirs(save_dir, exist_ok=True)

                fig_before.savefig(os.path.join(save_dir, f"raw_sub-{subject}_{label}-{item}_before.png"))
                fig_after.savefig(os.path.join(save_dir, f"raw_sub-{subject}_{label}-{item}_after.png"))
                plt.close('all')  # Close figures to avoid memory leaks


        ##----------------------------------------------------------------------------##
        #                     4. EPOCH THE CLEAN RAW FOR ANALYSIS                      #
        ##----------------------------------------------------------------------------##
        ''' Epoch the ICA-cleaned raw based on experimental events and apply linear
            detrending to preserve frequency information while removing slow drifts
            and removing DC offset with baseline correction. The epochs are created 
            with a length defined in the config file. 

            RESULTS:
            - Analysis-ready epochs
        '''
        epochs_dict = create_analysis_epochs(
            raw=raw,
            dataset=dataset, 
            eeg_settings=EEG_SETTINGS, 
            subject=subject, 
            item=item, 
            verbose=True
        )

        epochs_merged = mne.concatenate_epochs(list(epochs_dict.values()))

        # Get rejection threshold with bayesion optimization
        reject = get_scaled_rejection_threshold(epochs_merged, reject_scale_factor=1.5)
        print(f"Reject threshold for subject-{subject} {label}-{item}: {reject}")
        log_reject_threshold(
            reject=reject,
            dataset=dataset,
            subject=subject,
            stage="analysis_epochs",
            **kwargs, 
        )

        # Save full copy before dropping
        original_epochs = epochs_merged.copy()
        epochs_merged.drop_bad(reject, verbose=True)

        # Saving epochs
        save_epochs_dict(epochs_dict, reject, dataset, subject, item)

        log_dropped_epochs(
            epochs=epochs_merged, 
            dataset=dataset,
            subject=subject,
            stage="analysis_epochs",
            threshold=reject['eeg'],
            **kwargs
        )

        # Find dropped epoch indices
        dropped_indices = get_dropped_epoch_indices(epochs_merged.drop_log)
        # dropped_epochs = original_epochs[dropped_indices]   # <-- this line is not working due to unknown reason, but whatever...

        if not dropped_indices:
            print(f"[INFO] No dropped epochs for sub-{subject}, {label}-{item}")
            continue

        # create evoked objects of epochs before and after dropping
        evoked_before = original_epochs.average()
        evoked_after = epochs_merged.average()

        # Plot the evoked objects
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        ylim_from_data = max_ptp_evoked(evoked_before, evoked_after)*1e6
        ylim = dict(eeg=(-ylim_from_data, ylim_from_data))
        evoked_before.plot(exclude=[], axes=axes[0], ylim=ylim, show=False, time_unit='s')
        axes[0].set_title('Evoked before reject')
        evoked_after.plot(exclude=[], axes=axes[1], ylim=ylim, show=False, time_unit='s')
        axes[1].set_title('Evoked after reject')
        plt.tight_layout()
        # plt.suptitle(f"Evoked | Subject: {subject} | {label}: {item}", fontsize=12)
        save_path = os.path.join(
            PLOTS_PATH, dataset.f_name, "bad_analysis_epochs_evoked", f"evoked_{subject}_{label}_{item}.png"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

        # clear the figure
        plt.clf()
        plt.close()

        # Create psd before and after dropping
        plot_overlapping_psd(
            epochs_before=original_epochs,
            epochs_after=epochs_merged,
            title=f"PSD Comparison: Subject {subject}",
            save_path=os.path.join(PLOTS_PATH, dataset.f_name, "bad_analysis_epochs_psd", f"subject-{subject}_{label}-{item}.png")
        )






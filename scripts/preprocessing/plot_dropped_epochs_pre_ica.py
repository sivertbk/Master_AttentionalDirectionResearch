import os
import mne
import json
from utils.config import DATASETS, EEG_SETTINGS, PREPROCESSING_LOG_PATH
from utils.file_io import load_raw_data
from utils.helpers import iterate_dataset_items
from tqdm import tqdm


def load_reject_threshold(dataset, subject, session=None, task=None, run=None, stage="pre_ica"):
    """Load the reject threshold for a specific subject from JSON."""
    fname = f"{stage}_reject_thresholds.json"
    path = os.path.join(PREPROCESSING_LOG_PATH, "reject_thresholds", dataset.f_name, fname)

    if not os.path.exists(path):
        print(f"[WARN] No threshold file found: {path}")
        return None

    with open(path, "r") as f:
        all_thresholds = json.load(f)

    key = f"sub-{subject}"
    if session is not None:
        key += f"_ses-{session}"
    if task is not None:
        key += f"_task-{task}"
    if run is not None:
        key += f"_run-{run}"

    return all_thresholds.get(key, None)


def get_dropped_epoch_indices(drop_log):
    return [i for i, log in enumerate(drop_log) if len(log) > 0]


def plot_dropped_epochs(dropped_epochs, subject, label, item):
    title = f"Dropped Epochs | Subject: {subject} | {label}: {item}"
    dropped_epochs.plot(block=True, scalings=dict(eeg=150e-6), title=title)


if __name__ == "__main__":
    for dataset, subject, label, item, kwargs in iterate_dataset_items(DATASETS):
        raw = load_raw_data(dataset, subject, **kwargs, verbose=True)
        if raw is None:
            continue

        # Reconstruct preprocessing up to epochs (without dropping)
        from utils.preprocessing_tools import prepare_raw_data, fix_bad_channels

        raw = prepare_raw_data(raw, dataset, EEG_SETTINGS)
        raw = fix_bad_channels(raw, dataset, subject=subject, **kwargs, verbose=True)
        raw.filter(l_freq=EEG_SETTINGS['LOW_CUTOFF_HZ'], h_freq=None, verbose=True)

        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=EEG_SETTINGS["SYNTHETIC_LENGTH"],
            preload=True,
            verbose=True
        )
        epochs.pick(picks=["eeg", "eog"])
        epochs.set_eeg_reference(ref_channels="average", projection=False)

        # Load previously used threshold
        reject = load_reject_threshold(dataset, subject, **kwargs)
        if reject is None:
            print(f"[SKIP] No reject threshold for subject {subject}, skipping.")
            continue

        # Save full copy before dropping
        full_epochs = epochs.copy()
        epochs.drop_bad(reject, verbose=True)

        # Find dropped epoch indices
        dropped_indices = get_dropped_epoch_indices(epochs.drop_log)

        if not dropped_indices:
            print(f"[INFO] No dropped epochs for sub-{subject}, {label}-{item}")
            continue

        # Extract and plot the dropped epochs
        dropped_epochs = full_epochs[dropped_indices]
        plot_dropped_epochs(dropped_epochs, subject, label, item)

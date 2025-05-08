from tqdm import tqdm

from utils.helpers import iterate_dataset_items
from utils.preprocessing_tools import prepare_raw_data, ransac_detect_bad_channels
from utils.file_io import load_raw_data
from utils.config import DATASETS, set_plot_style, EEG_SETTINGS

set_plot_style()

for dataset, subject, label, item, kwargs in iterate_dataset_items(DATASETS):
    raw = load_raw_data(dataset, subject, **kwargs)
    if raw is None:
        tqdm.write(f"    No data found for subject {subject} with {label} {item}.")
        continue

    raw = prepare_raw_data(raw, dataset, EEG_SETTINGS)

    ransac_detect_bad_channels(
        raw,
        dataset,
        EEG_SETTINGS,
        subject,
        **kwargs,
        verbose=True,
        save_plot=True,
        show_plot=False
    )
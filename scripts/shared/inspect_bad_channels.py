import os
import matplotlib.pyplot as plt

from utils.helpers import cleanup_memory, iterate_dataset_items
from utils.file_io import load_raw_data, load_bad_channels, update_bad_channels_json
from utils.preprocessing_tools import prepare_raw_data
from utils.config import DATASETS, set_plot_style, EEG_SETTINGS
import utils.config as config

set_plot_style()

DATASETS['braboszcz2017'].subjects = ['001','002']
DATASETS['jin2019'].subjects = []
DATASETS['touryan2022'].subjects = ['01', '02']


for dataset, subject, label, item, kwargs in iterate_dataset_items(DATASETS):
    bad_chs = load_bad_channels(
            save_dir=dataset.path_derivatives,
            dataset=dataset.f_name,
            subject=subject,
            **kwargs
        )
    
    if bad_chs is None:
        print(f"    No bad channels found for subject {subject} with {label} {item}.")
        continue

    # Load data for the current subject and iteration
    raw = load_raw_data(dataset, subject, **kwargs)
    if raw is None:
        print(f"    No data found for subject {subject} with {label} {item}.")
        continue
    # Prepare the raw data
    raw = prepare_raw_data(raw, dataset, EEG_SETTINGS)

    raw.pick_types(eeg=True)

    raw.info['bads'] = bad_chs

    raw.plot(highpass=1, lowpass=None, block=True, scalings=dict(eeg=100e-6))

    bad_chs = raw.info['bads']

    update_bad_channels_json(
            save_dir=dataset.path_derivatives,
            bad_chs=bad_chs,
            subject=subject,
            dataset=dataset.f_name,
            mode="inspect",
            **kwargs
            )
    
    path_plots = os.path.join(config.PLOTS_PATH, dataset.f_name)
    os.makedirs(path_plots, exist_ok=True)
    
    # plot the bad channels
    raw.info['bads'] = bad_chs

    fig = raw.plot_sensors(show_names=True, kind="topomap", show=False)
    title = f"Interpolated Bad Channels | Dataset: {dataset.name} | Subject: {subject} | {label}: {item}"
    fig.suptitle(title)
    fig.savefig(os.path.join(path_plots, "bad_chans", f"sub-{subject}_ses-{item}_sensors.png"))
    plt.close(fig)

    # Cleanup memory
    cleanup_memory('raw')
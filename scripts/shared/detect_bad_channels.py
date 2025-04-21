from utils.preprocessing_tools import prepare_raw_data, ransac_detect_bad_channels
from utils.file_io import load_raw_data

from utils.config import DATASETS, set_plot_style, EEG_SETTINGS

set_plot_style()

session = None
task = None
run = None

for dataset in DATASETS.values():
    print(f"Processing dataset: {dataset.name}")
    iter_list= []
    if dataset.f_name == "braboszcz2017":
        tasks = dataset.tasks
        sessions = None
        runs = None
    elif dataset.f_name == "jin2019":
        tasks = None
        sessions = dataset.sessions
        runs = None
    elif dataset.f_name == "touryan2022":
        tasks = None
        sessions = None
        runs = dataset.runs

    # Loop through subjects in the dataset
    for subject in dataset.subjects:
        print(f"  Subject: {subject}")

        # Determine what to iterate over based on the dataset configuration
        iteration_items = []
        if sessions is not None:
            iteration_items.append(('session', sessions))
        if tasks is not None:
            iteration_items.append(('task', tasks))
        if runs is not None:
            iteration_items.append(('run', runs))

        # Iterate through the valid variables (sessions, tasks, or runs)
        for label, iter_list in iteration_items:
            print(f"    Iteration: {label}")
            for item in iter_list:
                # Dynamically construct kwargs based on the dataset configuration
                kwargs = {label: item}

                # Load data for the current subject and iteration
                raw = load_raw_data(dataset, subject, **kwargs)
                if raw is None:
                    print(f"    No data found for subject {subject} with {label} {item}.")
                    continue
                # Prepare the raw data
                raw = prepare_raw_data(raw, dataset, EEG_SETTINGS)

                ransac_detect_bad_channels(raw, dataset, EEG_SETTINGS, subject, **kwargs, verbose=True, save_plot=True, show_plot=False)
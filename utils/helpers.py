import os
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
from mne.preprocessing import ICA
import gc
import datetime
import json
from itertools import combinations
from mne.channels import find_ch_adjacency
from mne.viz import plot_ch_adjacency

def print_directory_tree(
    root_dir=None,
    indent="",
    file_limit=10,
    dir_limit=10,
    exclude_dirs=None,
    max_depth=None,
    current_depth=0,
    skip_folders=None,
    output_lines=None,
    save_to_file=False,
    filename="directory_structure.txt"
):
    """
    Prints or saves a directory tree from the root_dir with various options.

    Args:
        root_dir (str): Starting directory (defaults to current working directory).
        indent (str): Indentation string for pretty print.
        file_limit (int): Max number of files per folder to show.
        dir_limit (int): Max number of folders per folder to show.
        exclude_dirs (list): List of dir names to exclude completely.
        max_depth (int): Maximum recursion depth.
        current_depth (int): Current level (used internally for recursion).
        skip_folders (list): List of folder names to skip diving into.
        output_lines (list): Internal list of lines being built.
        save_to_file (bool): If True, writes to file instead of printing.
        filename (str): Filename to save the output (if save_to_file=True).
    """
    if root_dir is None:
        root_dir = os.getcwd()

    if exclude_dirs is None:
        exclude_dirs = [".git", "__pycache__", ".DS_Store", "Thumbs.db", "attentional_direction_research_workspace.egg-info"]

    if skip_folders is None:
        skip_folders = ["raw", "psd_data"]

    if output_lines is None:
        output_lines = []

    if max_depth is not None and current_depth > max_depth:
        output_lines.append(f"{indent}... (max depth reached)")
        return output_lines

    try:
        entries = sorted(os.listdir(root_dir))
    except PermissionError:
        output_lines.append(f"{indent}🔒 [Access Denied]")
        return output_lines

    dirs = [e for e in entries if os.path.isdir(os.path.join(root_dir, e)) and e not in exclude_dirs]
    files = [e for e in entries if os.path.isfile(os.path.join(root_dir, e))]

    display_dirs = dirs[:dir_limit]
    remaining_dirs = len(dirs) - len(display_dirs)

    display_files = files[:file_limit]
    remaining_files = len(files) - len(display_files)

    for d in display_dirs:
        output_lines.append(f"{indent}📂 {d}/")
        full_path = os.path.join(root_dir, d)
        if d in skip_folders:
            output_lines.append(f"{indent}│   ... (skipped)")
        else:
            print_directory_tree(
                full_path,
                indent + "│   ",
                file_limit,
                dir_limit,
                exclude_dirs,
                max_depth,
                current_depth + 1,
                skip_folders,
                output_lines,
                save_to_file,
                filename
            )

    if remaining_dirs > 0:
        output_lines.append(f"{indent}... ({remaining_dirs} more directories)")

    for f in display_files:
        output_lines.append(f"{indent}📄 {f}")

    if remaining_files > 0:
        output_lines.append(f"{indent}... ({remaining_files} more files)")

    # Final output
    if current_depth == 0:
        if save_to_file:
            with open(filename, "w") as f:
                f.write("\n".join(output_lines))
            print(f"✅ Directory structure saved to: {filename}")
        else:
            for line in output_lines:
                print(line)

    return output_lines

def perform_ica_cleaning(epochs, subject, n_components=0.99, method='infomax', random_state=97):
    """
    Perform ICA on the provided epochs and allow manual selection of components for exclusion.

    Parameters:
        epochs (mne.Epochs): The epochs object to clean.
        n_components (int): Number of ICA components to compute (default: 0.99 - 99% of variance).
        method (str): ICA method to use ('infomax' by default).
        random_state (int): Random seed for ICA initialization (default: 97).

    Returns:
        mne.Epochs: Cleaned epochs after ICA.
        ICA: The ICA object used for the analysis.
    """
    print("Fitting ICA...")
    # Initialize ICA
    ica = ICA(n_components=n_components, method=method, random_state=random_state)

    # Fit ICA to the epochs
    ica.fit(epochs)
    print("ICA fitting complete.")

    # Plot ICA components
    print("Displaying ICA components. Use the interactive GUI to select components for exclusion.")
    ica.plot_components(title=f"ICA decomposition on subject {subject}") 

    # Plot sources to identify artifacts
    print("Displaying ICA sources. Use this to examine the time-series for artifact-related components.")
    ica.plot_sources(epochs, block = True)  # Opens an interactive source plot

    # Apply ICA to remove the selected components
    print("Applying ICA to remove selected components...")
    cleaned_epochs = ica.apply(epochs.copy())
    print("ICA cleaning complete.")

    return cleaned_epochs, ica

def format_numbers(numbers, width=3):
    """
    Format a number or a list of numbers to strings with leading zeros.

    Parameters:
        numbers (int, float, str, list): The number(s) to format.
        width (int): The width of the formatted string (default: 3).

    Returns:
        str or list: Formatted number(s) as string(s).
    """
    if isinstance(numbers, (int, float)):
        return f"{numbers:0{width}d}"
    elif isinstance(numbers, str):
        return f"{int(numbers):0{width}d}"
    elif isinstance(numbers, list):
        return [f"{int(num):0{width}d}" for num in numbers]
    else:
        raise TypeError("Input must be an int, float, str, or list of these types.")

def calculate_freq_resolution(sampling_rate, n_fft):
    """
    Calculate frequency resolution based on sampling rate and FFT size.

    Parameters:
        sampling_rate (float): Sampling rate in Hz.
        n_fft (int): Number of FFT points.

    Returns:
        float: Frequency resolution in Hz.
    """
    return sampling_rate / n_fft


def calculate_rayleigh_freq_resolution(epoch_length_sec):
    """
    Calculate frequency resolution (Rayleigh frequency).

    Parameters:
        epoch_length_sec (float): Length of epoch in seconds.

    Returns:
        float: Frequency resolution in Hz.
    """
    return 1.0 / epoch_length_sec

def find_class(event_code, class_dict):
    """
    Find and return the class name corresponding to a given event code.

    Parameters:
        event_code (int): The numeric event code to classify.
        class_dict (dict): A dictionary mapping class names (str) to lists of event codes (list[int]).

    Returns:
        str: The corresponding class name if the event code is found; otherwise, returns 'undefined'.
    """
    for class_name, codes in class_dict.items():
        if event_code in codes:
            return class_name
    return 'undefined'


def get_class_array(epochs, class_dict):
    """
    Generate an array of class labels for each epoch in an MNE Epochs object.

    Parameters:
        epochs (mne.Epochs): The epochs object containing event information.
        class_dict (dict): A dictionary mapping class names (str) to lists of event codes (list[int]).

    Returns:
        list[str]: List of class labels (one per epoch) corresponding to the epochs' event codes.
    """
    tasks = [find_class(code, class_dict) for code in epochs.events[:, 2]]
    return tasks


def generate_metadata_df(epochs, dataset_config, subject, session, eeg_settings, psd_data):
    """
    Generate comprehensive metadata for epochs and PSD configuration.

    Parameters:
        epochs (mne.Epochs): EEG epochs data.
        dataset_config (DatasetConfig): Configuration object for dataset.
        subject (str/int): Subject identifier.
        session (str/int): Session identifier.
        eeg_settings (dict): Dictionary containing EEG and PSD settings.
        psd_data (np.ndarray): Computed PSD array (epochs x channels x frequencies).

    Returns:
        pd.DataFrame: Metadata DataFrame per epoch.
    """
    # Basic metadata from configuration
    epoch_metadata = pd.DataFrame({
        'dataset_name': dataset_config.name,
        'subject': subject,
        'session': session,
        'state': get_class_array(epochs, dataset_config.state_classes),
        'task': get_class_array(epochs, dataset_config.task_classes),
        'task_orientation': dataset_config.task_orientation
    })

    # Add subject group if available
    if 'subject_groups' in dataset_config.extra_info:
        epoch_metadata['subject_group'] = dataset_config.extra_info["subject_groups"].get(subject, "NA")

    # PSD-specific metadata (for reproducibility)
    unit_suffix = 'µV²/Hz' if eeg_settings["PSD_UNIT_CONVERT"] == 1e12 else 'V²/Hz'
    epoch_metadata['psd_units'] = unit_suffix
    epoch_metadata['sampling_rate_hz'] = eeg_settings["SAMPLING_RATE"]
    epoch_metadata['epoch_duration_sec'] = eeg_settings["EPOCH_LENGTH_SEC"]
    epoch_metadata['psd_method'] = "Welch"
    epoch_metadata['psd_n_per_seg'] = eeg_settings["PSD_N_PER_SEG"]
    epoch_metadata['psd_n_overlap'] = eeg_settings["PSD_N_OVERLAP"]
    epoch_metadata['psd_n_fft'] = eeg_settings["PSD_N_FFT"]
    epoch_metadata['psd_freq_resolution_hz'] = eeg_settings["SAMPLING_RATE"] / eeg_settings["PSD_N_FFT"]
    epoch_metadata['psd_average_method'] = eeg_settings["PSD_AVERAGE_METHOD"]
    epoch_metadata['psd_freq_range_hz'] = f"{eeg_settings['PSD_FMIN']}-{eeg_settings['PSD_FMAX']} Hz"
    epoch_metadata['psd_shape'] = [psd_data.shape for _ in range(len(epoch_metadata))]
    epoch_metadata['psd_computed_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return epoch_metadata

def generate_metadata_epochs(epochs, dataset_config, subject, session):
    return pd.DataFrame({
        'dataset_name': dataset_config.name,
        'subject': subject,
        'session': session,
        'state': get_class_array(epochs, dataset_config.state_classes),
        'task': get_class_array(epochs, dataset_config.task_classes),
        'task_orientation': dataset_config.task_orientation,
        'subject_group': dataset_config.extra_info.get("subject_groups", {}).get(subject, "NA")
    })

def generate_metadata_config(eeg_settings, psd_data):
    return {
        "sampling_rate": eeg_settings["SAMPLING_RATE"],
        "epoch_duration_sec": eeg_settings["EPOCH_LENGTH_SEC"],
        "psd_method": "Welch",
        "psd_window": eeg_settings["PSD_WINDOW"],
        "psd_output": eeg_settings["PSD_OUTPUT"],
        "psd_units": "µV²/Hz" if eeg_settings["PSD_UNIT_CONVERT"] == 1e12 else "V²/Hz",
        "psd_n_fft": eeg_settings["PSD_N_FFT"],
        "psd_n_per_seg": eeg_settings["PSD_N_PER_SEG"],
        "psd_n_overlap": eeg_settings["PSD_N_OVERLAP"],
        "psd_freq_resolution_hz": eeg_settings["SAMPLING_RATE"] / eeg_settings["PSD_N_FFT"],
        "psd_average_method": eeg_settings["PSD_AVERAGE_METHOD"],
        "psd_freq_range_hz": f"{eeg_settings['PSD_FMIN']}-{eeg_settings['PSD_FMAX']} Hz",
        "psd_shape": list(psd_data.shape),
        "psd_remove_dc": eeg_settings["PSD_REMOVE_DC"],
        "psd_computed_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def remap_events_to_original(epochs):
    """
    Remap simplified event codes (1, 2, 3, 4) back to original event codes explicitly.

    Parameters:
        epochs (mne.Epochs): Epochs object with simplified event codes.
        dataset_config (DatasetConfig): Dataset configuration containing original event_classes.

    Returns:
        epochs (mne.Epochs): Same epochs object with updated events and event_id.
    """
    simplified_event_id = {
        'vs/MW': 1, 
        'sart/MW': 2, 
        'vs/OT': 3, 
        'sart/OT': 4
    }

    # Invert simplified mapping clearly
    simplified_id_to_class = {v: k for k, v in simplified_event_id.items()}

    # Choose representative original event codes explicitly
    representative_original_codes = { 
        'vs/MW': 103,  
        'sart/MW': 113, 
        'vs/OT': 101,   
        'sart/OT': 111  
    }

    # Remap epochs.events in place explicitly
    for simplified_id, class_label in simplified_id_to_class.items():
        original_code = representative_original_codes[class_label]
        epochs.events[epochs.events[:, 2] == simplified_id, 2] = original_code

    # Update event_id explicitly
    epochs.event_id = representative_original_codes

    return epochs

def get_available_variants(path_psd, subject_id, session_id):
    """
    List all available PSD variants for a given subject-session.

    Parameters:
        path_psd (str): Root path to the psd_data directory (e.g., dataset_config.path_psd).
        subject_id (str or int): Subject identifier (e.g., '001').
        session_id (str or int): Session identifier (e.g., '1').

    Returns:
        List[str]: List of available variant names (e.g., ['avg-mean', 'avg-median']).
    """
    sub_dir = os.path.join(path_psd, f"sub-{subject_id}", f"ses-{session_id}")
    
    if not os.path.isdir(sub_dir):
        return []

    variants = [
        name for name in os.listdir(sub_dir)
        if os.path.isdir(os.path.join(sub_dir, name))
    ]

    return sorted(variants)

def cleanup_memory(*var_names):
    """
    Deletes variables from the caller's local scope (if they exist) and runs garbage collection.

    Parameters
    ----------
    var_names : str
        Names of variables to delete as strings.
    """
    import psutil, gc, datetime, inspect

    process = psutil.Process()
    print(f"[{datetime.datetime.now()}] Memory before cleanup: {process.memory_info().rss / 1e6:.2f} MB")

    caller_locals = inspect.currentframe().f_back.f_locals

    for name in var_names:
        if name in caller_locals:
            try:
                del caller_locals[name]
            except Exception as e:
                print(f"⚠️ Could not delete {name}: {e}")

    gc.collect()
    print(f"[{datetime.datetime.now()}] Memory after cleanup: {process.memory_info().rss / 1e6:.2f} MB")
    
def plot_ransac_bad_log(ransac, epochs_hp, subject_id, dataset, meta_info, save_path=None, show=True):
    """
    Plot RANSAC bad_log heatmap with channels on Y-axis and trials on X-axis.
    Highlights bad channels in red and titles the plot with dataset metadata.

    Parameters:
    - ransac: fitted autoreject.Ransac object
    - epochs_hp: MNE Epochs object used with RANSAC
    - subject_id: str or int, e.g. "001"
    - dataset: str, one of "jin2019", "braboszcz2017", "touryan2022"
    - meta_info: str/int describing session/task/run depending on dataset
    - save_path: str or Path or None. If provided, saves the figure to this path.
    """

    # Transpose bad_log: shape becomes (channels, trials)
    bad_log_T = ransac.bad_log.T

    # Get channel names for picked indices
    picked_ch_names = [epochs_hp.ch_names[ii] for ii in ransac.picks]

    # Which channels were flagged as bad overall
    bad_chs = ransac.bad_chs_

    # Title part
    label_type = {
        "Jin et al. (2019)": "Session",
        "Braboszcz et al. (2017)": "Task",
        "Touryan e tal. (2022)": "Run"
    }.get(dataset, "Meta")

    title = f"Bad channels detected with RANSAC  |  Dataset: {dataset} | Subject: {subject_id} | {label_type}: {meta_info}"

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
    im = ax.imshow(bad_log_T, cmap='Reds', interpolation='nearest', aspect='auto')

    ax.grid(False)
    ax.set_xlabel('Trials')
    ax.set_ylabel('Sensors')
    ax.set_title(title, fontsize=13)

    # Set tick labels
    yticks = np.arange(len(picked_ch_names))
    ax.set_yticks(yticks)
    ax.set_yticklabels(picked_ch_names)

    xticks = np.arange(bad_log_T.shape[1])
    ax.set_xticks(xticks[::max(1, len(xticks)//20)])  # avoid overcrowding
    ax.set_xticklabels(xticks[::max(1, len(xticks)//20)])

    # Make bad channels red in y-axis tick labels
    for tick_label, ch_name in zip(ax.get_yticklabels(), picked_ch_names):
        if ch_name in bad_chs:
            tick_label.set_color('red')

    # Save if path is provided
    if save_path is not None:
        save_path = str(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[RANSAC Plot] Saved to: {save_path}")

    if show:
        plt.show()
    
    else:
        plt.close(fig)  # Prevents figure from being displayed later


def get_largest_strict_interp_cluster_sizes(reject_log, info, ch_type='eeg', verbose=True):
    """
    Compute the size of the largest strictly connected cluster of interpolated channels per epoch.
    A strict cluster starts with a fully connected triangle (3 nodes all mutually connected),
    and grows only by adding nodes that are connected to at least 2 nodes already in the cluster.

    Parameters:
    - reject_log: instance of autoreject.RejectLog
    - info: instance of mne.Info
    - ch_type: str, channel type to consider (default: 'eeg')

    Returns:
    - cluster_sizes: np.ndarray of shape (n_epochs,)
    """
    interp_mask = reject_log.labels == 2  # Interpolated channels
    adjacency, ch_names = find_ch_adjacency(info, ch_type=ch_type)

    if verbose:
        print(f"Adjacency matrix shape: {adjacency.shape}")
        print(f"Interpolated channels shape: {interp_mask.shape}")
        plot_ch_adjacency(info=info, adjacency=adjacency, ch_names=ch_names)

    adjacency = adjacency.toarray()

    cluster_sizes = []

    for epoch_idx in range(interp_mask.shape[0]):
        bad_chs = np.where(interp_mask[epoch_idx])[0]
        if len(bad_chs) < 3:
            cluster_sizes.append(len(bad_chs))  # No triangle possible
            continue

        sub_adj = adjacency[np.ix_(bad_chs, bad_chs)]
        max_cluster = 0

        for combo in combinations(range(len(bad_chs)), 3):
            i, j, k = combo
            if sub_adj[i, j] and sub_adj[i, k] and sub_adj[j, k]:
                cluster = set([i, j, k])
                changed = True
                while changed:
                    changed = False
                    for n in range(len(bad_chs)):
                        if n in cluster:
                            continue
                        neighbors = [m for m in cluster if sub_adj[n, m]]
                        if len(neighbors) >= 2:
                            cluster.add(n)
                            changed = True
                max_cluster = max(max_cluster, len(cluster))

        cluster_sizes.append(max_cluster if max_cluster > 0 else 1)

    return np.array(cluster_sizes)

def reject_epochs_with_adjacent_interpolation(epochs, reject_log, max_cluster_size=3, ch_type='eeg', verbose=True):
    """
    Reject epochs with interpolated channel clusters larger than max_cluster_size.

    Parameters:
    - epochs: mne.Epochs object (original or already AutoReject-cleaned)
    - reject_log: instance of autoreject.RejectLog
    - max_cluster_size: int, maximum allowed size of adjacent interpolation cluster
    - ch_type: str, channel type to use for adjacency (default: 'eeg')

    Returns:
    - cleaned_epochs: mne.Epochs object with offending epochs dropped
    - rejected_indices: list of rejected epoch indices
    """
    cluster_sizes = get_largest_strict_interp_cluster_sizes(reject_log, epochs.info, ch_type=ch_type, verbose=verbose)
    if verbose:
        print(f"Cluster sizes: {cluster_sizes}")
    reject_mask = cluster_sizes > max_cluster_size
    cleaned_epochs = epochs.copy()[~reject_mask]
    rejected_indices = np.where(reject_mask)[0].tolist()
    return cleaned_epochs, rejected_indices

def update_bad_epochs_from_indices(reject_log, rejected_idxs, verbose=True):
    """
    Update the bad_epochs array of a RejectLog using a list of rejected indices.

    Parameters:
    - reject_log: autoreject.RejectLog instance
    - rejected_idxs: list or array of ints (indices to set as bad)
    """
    new_bad_epochs = np.zeros_like(reject_log.bad_epochs, dtype=bool)
    new_bad_epochs[rejected_idxs] = True
    reject_log.bad_epochs = new_bad_epochs
    if verbose:
        print(f"Updated reject_log.bad_epochs with {len(rejected_idxs)} rejected epochs.")


def plot_rejected_epochs_by_cluster(epochs, rejected_indices, ch_type="eeg", n_epochs_to_plot=None):
    """
    Plot rejected epochs due to spatially adjacent interpolations.

    Parameters:
    - epochs: mne.Epochs object (should contain the rejected epochs)
    - rejected_indices: list of indices for rejected epochs
    - ch_type: str, channel type to plot (default: 'eeg')
    - n_epochs_to_plot: int or None, how many rejected epochs to plot (default: all)
    """
    if not rejected_indices:
        print("No rejected epochs to plot.")
        return

    if n_epochs_to_plot is None:
        plot_indices = rejected_indices
    else:
        plot_indices = rejected_indices[:n_epochs_to_plot]

    fig = plt.figure(figsize=(12, 2.5 * len(plot_indices)))
    for i, idx in enumerate(plot_indices):
        ax = fig.add_subplot(len(plot_indices), 1, i + 1)
        epochs[idx].plot(picks=ch_type, axes=ax, show=False, title=f"Rejected Epoch {idx}")
    plt.tight_layout()
    plt.show()

def plot_dropped_epochs_by_cluster(epochs, reject_log, rejected_indices, scalings=dict(eeg=2e-3), title="Dropped epochs due to adjacent interpolation"):
    """
    Plot epochs that were rejected due to clustered interpolation, styled similarly to RejectLog.plot_epochs.

    Parameters:
    - epochs: mne.Epochs object (should include the rejected epochs)
    - reject_log: instance of autoreject.RejectLog
    - rejected_indices: list of indices of rejected epochs
    - scalings: dict or None, passed to epochs.plot
    - title: str, plot title

    Returns:
    - fig: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from mne.viz import plot_epochs as plot_mne_epochs

    labels = reject_log.labels
    color_map = {0: 'k', 1: 'r', 2: 'b'}
    epoch_colors = []

    for idx in rejected_indices:
        label_epoch = labels[idx]
        color_row = [
            color_map[int(lbl)] if not np.isnan(lbl) else 'k'
            for lbl in label_epoch
        ]
        epoch_colors.append(color_row)

    epochs_subset = epochs[rejected_indices]
    fig = plot_mne_epochs(
        epochs=epochs_subset,
        events=epochs_subset.events,
        epoch_colors=epoch_colors,
        scalings=scalings,
        title=title
    )
    return fig


if __name__ == "__main__":
    print("Helper functions loaded successfully.")
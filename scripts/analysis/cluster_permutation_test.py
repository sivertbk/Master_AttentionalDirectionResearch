import numpy as np
import pandas as pd
import os
from mne.channels import find_ch_adjacency
from mne.stats import permutation_cluster_1samp_test
from mne import create_info
from mne.viz import plot_topomap
import matplotlib.pyplot as plt

from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import DATASETS, set_plot_style

set_plot_style()  # Set the plotting style for matplotlib

def build_X(df: pd.DataFrame, task_orientation: str) -> np.ndarray:
    """
    Builds a np.array with the mean log band power state difference for each subject
    for the specified task orientation.
    """
    df = df[df['task_orientation'] == task_orientation]
    return np.stack(df['mean_diff'].values) # shape: (n_subjects, n_channels)
    

def perform_cluster_permutation_test(X: np.ndarray, adjacency, tail, n_permutations=10000):
    """
    Perform a cluster-based permutation test.

    Parameters:
    - X: The data array to test.
    - adjacency: The adjacency matrix for the channels.
    - tail: The tail of the test (1 or 2).
    - n_permutations: The number of permutations to perform.

    Returns:
    - p_values: The p-values from the permutation test.
    """
    return permutation_cluster_1samp_test(
        X,
        n_permutations=n_permutations,
        threshold=None,
        tail=tail,
        adjacency=adjacency,
        seed=42,
        out_type='mask'
    )

def get_significant_mask(clusters, pvals, alpha=0.05):
    """
    Return the combined mask of significant clusters if any, or the strongest non-significant cluster.
    Also returns a dictionary with plotting style (e.g., markerfacecolor).
    """
    if len(clusters) == 0:
        return None, {}

    # Find all significant clusters
    significant_mask = np.zeros_like(clusters[0], dtype=bool)
    found_significant = False
    for cluster, p in zip(clusters, pvals):
        if p < alpha:
            significant_mask |= cluster
            found_significant = True

    if found_significant:
        # Use black marker (default) to highlight significant channels
        style = dict(marker='o', markerfacecolor='k', markeredgecolor='k',
                     linewidth=0, markersize=4)
        return significant_mask, style
    else:
        # Find best non-significant cluster (lowest p)
        best_idx = int(np.argmin(pvals))
        best_mask = clusters[best_idx]

        # Use red marker to indicate this is a non-significant cluster
        style = dict(marker='o', markerfacecolor='red', markeredgecolor='red',
                     linewidth=0, markersize=4)
        return best_mask, style

def plot_task_orientation_map(T_vals, mask, info, title, topomap_args=None):
    if topomap_args is None:
        topomap_args = dict(marker='o', markerfacecolor='k', markeredgecolor='k',
                            linewidth=0, markersize=4)

    fig, ax = plt.subplots()
    im, cn = plot_topomap(
        T_vals,
        pos=info,
        mask=mask,
        mask_params=topomap_args,
        cmap="RdBu_r",
        contours=0,
        show=False,
        axes=ax,
        sphere=0.1
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label("t-value", fontsize=10)

    # Add title
    ax.set_title(title)
    plt.tight_layout()

    # Save the figure as .svg in the analyzer derivatives folder
    output_dir = os.path.join(analyzer.derivatives_path, "cluster_test_results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{title.replace(' ', '_')}.svg")
    fig.savefig(output_path, format="svg")
    plt.close(fig)

if __name__ == "__main__":

    ANALYZER_NAME = "eeg_analyzer_test"

    # Trying to load the EEGAnalyzer
    analyzer = EEGAnalyzer.load_analyzer(ANALYZER_NAME)
    if analyzer is None:
        print(f"Analyzer {ANALYZER_NAME} not found. Creating a new one.")
        analyzer = EEGAnalyzer(DATASETS, ANALYZER_NAME)

    analyzer.save_analyzer()  # Ensure the analyzer is saved

    df = pd.DataFrame()

    for dataset in analyzer: 
        for subject in dataset: 
            for recording in subject:
                if recording.exclude:
                    print(f"Skipping excluded recording for subject {subject.id} in dataset {dataset.name}.")
                    continue
                # make structured dictionary for each recording
                # # This will be used to create a DataFrame later 
                mean_OT = recording.get_stat('mean',
                                            data_type='log_band_power',
                                            state='OT',
                                            filtered=True)   
                mean_MW = recording.get_stat('mean',
                                            data_type='log_band_power',
                                            state='MW',
                                            filtered=True)
                mean_diff = mean_OT - mean_MW
                data_dict = {
                    "dataset": dataset.name,
                    "subject": subject.id,
                    "group": subject.group,
                    "session": recording.session_id,
                    "task_orientation": dataset.task_orientation,
                    "mean_OT_alpha": mean_OT,
                    "mean_MW_alpha": mean_MW,
                    "mean_diff": mean_diff,
                }
                # Add the data_dict to the DataFrame
                df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)
                

    X_internal = build_X(df, task_orientation='internal')
    X_external = build_X(df, task_orientation='external')

    for dataset in analyzer:
        for subject in dataset:
            ch_names = subject.get_channel_names()
            if ch_names:
                break
        else:
            raise ValueError("No channels found in any subject.")
        
    info = create_info(ch_names, sfreq=1, ch_types='eeg') # sfreq is not used, but required
    info.set_montage(analyzer.get_montage())
    adjacency, ch_names = find_ch_adjacency(info, ch_type='eeg')

    # Perform cluster-based permutation test
    T_internal, clusters_int, pvals_int, _ = perform_cluster_permutation_test(X_internal, adjacency, tail=1)
    T_external, clusters_ext, pvals_ext, _ = perform_cluster_permutation_test(X_external, adjacency, tail=-1)

    # Get significant masks
    mask_internal, topomap_args_internal = get_significant_mask(clusters_int, pvals_int)
    mask_external, topomap_args_external = get_significant_mask(clusters_ext, pvals_ext)

    # Plotting the results
    if mask_internal is None or mask_external is None:
        print("No significant clusters found for internal or external task orientation.")
    else:
        plot_task_orientation_map(T_internal, mask_internal, info, "Alpha Power Difference (OT – MW) for Internally Oriented Tasks", topomap_args=topomap_args_internal)
        plot_task_orientation_map(T_external, mask_external, info, "Alpha Power Difference (OT – MW) for Externally Oriented Tasks", topomap_args=topomap_args_external)

    print("Internal cluster p-vals:", pvals_int)
    print("External cluster p-vals:", pvals_ext)

    # Internal reversed
    T_int_rev, clusters_int_rev, pvals_int_rev, _ = perform_cluster_permutation_test(X_internal, adjacency, tail=-1)
    # External reversed
    T_ext_rev, clusters_ext_rev, pvals_ext_rev, _ = perform_cluster_permutation_test(X_external, adjacency, tail=1)

    # Get significant masks for reversed tests
    mask_int_rev, topomap_args_int_rev = get_significant_mask(clusters_int_rev, pvals_int_rev)
    mask_ext_rev, topomap_args_ext_rev = get_significant_mask(clusters_ext_rev, pvals_ext_rev)

    if mask_int_rev is None or mask_ext_rev is None:
        print("No significant clusters found for reversed internal or external task orientation.")
    else:
        plot_task_orientation_map(T_int_rev, mask_int_rev, info, "Alpha Power Difference (OT – MW) for Internally Oriented Tasks", topomap_args=topomap_args_int_rev)
        plot_task_orientation_map(T_ext_rev, mask_ext_rev, info, "Alpha Power Difference (OT – MW) for Externally Oriented Tasks", topomap_args=topomap_args_ext_rev)

    print("Reversed internal cluster p-vals:", pvals_int_rev)
    print("Reversed external cluster p-vals:", pvals_ext_rev)

    # Create a new DataFrame with channel names as columns and rows for external and internal data
    mean_alpha_df = pd.DataFrame(
        {
            "Channel Names": info["ch_names"],
            "Internal": X_internal.mean(axis=0),
            "External": X_external.mean(axis=0),
        }
    )

    # Save the DataFrame to a CSV file
    output_dir = os.path.join(analyzer.derivatives_path, "cluster_test_results")
    os.makedirs(output_dir, exist_ok=True)
    mean_alpha_df.to_csv(os.path.join(output_dir, "mean_alpha_power.csv"), index=False)

    # Save or use the DataFrame as needed
    print(mean_alpha_df)
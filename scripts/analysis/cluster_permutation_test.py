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
    df_oriented = df[df['task_orientation'] == task_orientation]
    if df_oriented.empty:
        return np.array([])
    return np.stack(df_oriented['mean_diff'].values)  # shape: (n_subjects, n_channels)

def build_state_vectors(df: pd.DataFrame, task_orientation: str) -> np.ndarray:
    """
    Builds a np.array with the mean log band power for each subject
    for the specified task orientation. Returns both OT and MW states.
    The shape of the array is (n_subjects, n_channels).
    """
    df_oriented = df[df['task_orientation'] == task_orientation]
    if df_oriented.empty:
        return np.array([])
    return np.stack(df_oriented['mean_OT_alpha'].values), np.stack(df_oriented['mean_MW_alpha'].values)  # shape: (n_subjects, n_channels)

def perform_cluster_permutation_test(X: np.ndarray, adjacency, tail, n_permutations=10000):
    """
    Perform a cluster-based permutation test.
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
    Also returns a dictionary with plotting style (e.g., markerfacecolor) and the corresponding p-value.
    """
    if len(clusters) == 0:
        return None, {}, None

    significant_mask = np.zeros_like(clusters[0], dtype=bool)
    significant_pvals = []
    found_significant = False
    for cluster, p in zip(clusters, pvals):
        if p < alpha:
            significant_mask |= cluster
            significant_pvals.append(p)
            found_significant = True

    if found_significant:
        style = dict(marker='o', markerfacecolor='k', markeredgecolor='k',
                     linewidth=0, markersize=4)
        return significant_mask, style, np.min(significant_pvals)
    else:
        best_idx = int(np.argmin(pvals))
        best_mask = clusters[best_idx]
        best_pval = pvals[best_idx]
        style = dict(marker='o', markerfacecolor='red', markeredgecolor='red',
                     linewidth=0, markersize=4)
        return best_mask, style, best_pval


def plot_task_orientation_map(mean_alpha_power_OT, mean_alpha_power_MW, mean_alpha_power_diff, T_vals, mask, info, main_title, right_plot_title, output_dir, tail_name, topomap_args=None):
    """
    Plots and saves a figure with four topomaps in a single row:
    - mean alpha power OT,
    - mean alpha power MW,
    - mean alpha power difference (OT - MW),
    - and t-values.
    """
    if topomap_args is None:
        topomap_args = dict(marker='o', markerfacecolor='k', markeredgecolor='k',
                            linewidth=0, markersize=4)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(main_title, fontsize=16)

    # Common vlim for OT and MW plots
    vlim_min = np.min([mean_alpha_power_OT, mean_alpha_power_MW])
    vlim_max = np.max([mean_alpha_power_OT, mean_alpha_power_MW])

    # Plot 1: Mean Alpha Power On-Target
    im1, _ = plot_topomap(
        mean_alpha_power_OT,
        pos=info,
        cmap="RdBu_r",
        vlim=(vlim_min, vlim_max),
        contours=0,
        show=False,
        axes=axes[0],
        sphere=0.1
    )
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.7, pad=0.05)
    cbar1.set_label("ln(µV²)", fontsize=10)
    axes[0].set_title("Mean Alpha Power On-Target")

    # Plot 2: Mean Alpha Power Mind-Wandering
    im2, _ = plot_topomap(
        mean_alpha_power_MW,
        pos=info,
        cmap="RdBu_r",
        vlim=(vlim_min, vlim_max),
        contours=0,
        show=False,
        axes=axes[1],
        sphere=0.1
    )
    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.7, pad=0.05)
    cbar2.set_label("ln(µV²)", fontsize=10)
    axes[1].set_title("Mean Alpha Power Mind-Wandering")

    # Plot 3: Mean Alpha Power Difference
    vmax_diff = np.max(np.abs(mean_alpha_power_diff))
    vmin_diff = -vmax_diff
    im3, _ = plot_topomap(
        mean_alpha_power_diff,
        pos=info,
        cmap="RdBu_r",
        vlim=(vmin_diff, vmax_diff),
        contours=0,
        show=False,
        axes=axes[2],
        sphere=0.1
    )
    cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.7, pad=0.05)
    cbar3.set_label("ln(µV²)", fontsize=10)
    axes[2].set_title("Mean Alpha Power Difference (OT-MW)")

    # Plot 4: T-values
    im4, _ = plot_topomap(
        T_vals,
        pos=info,
        mask=mask,
        mask_params=topomap_args,
        cmap="RdBu_r",
        contours=0,
        show=False,
        axes=axes[3],
        sphere=0.1
    )
    cbar4 = plt.colorbar(im4, ax=axes[3], shrink=0.7, pad=0.05)
    cbar4.set_label("t-value", fontsize=10)
    axes[3].set_title(right_plot_title)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    sanitized_main_title = main_title.replace(' ', '_').replace('(', '').replace(')', '').replace('–', '-')
    output_filename = f"{sanitized_main_title}_{tail_name}.svg"
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def extract_dataframe(analyzer: EEGAnalyzer, skip_controls: bool, filter_task: bool, filtered: bool) -> pd.DataFrame:
    """
    Extracts mean log band power data from the analyzer into a DataFrame.
    """
    data_records = []
    for dataset in analyzer:
        for subject in dataset:
            if skip_controls and subject.group == 'ctr':
                print(f"Skipping control group subject {subject.id} in dataset {dataset.name}.")
                continue
            for recording in subject:
                if recording.exclude:
                    print(f"Skipping excluded recording for subject {subject.id} in dataset {dataset.name}.")
                    continue

                if filter_task and dataset.f_name == 'jin2019':
                    print(f"Filtering task-specific data for dataset {dataset.name} and subject {subject.id}.")
                    mean_OT = recording.get_stat('mean', data_type='log_band_power', state='OT', filtered=filtered, task='vs')
                    mean_MW = recording.get_stat('mean', data_type='log_band_power', state='MW', filtered=filtered, task='vs')
                else:
                    mean_OT = recording.get_stat('mean', data_type='log_band_power', state='OT', filtered=filtered)
                    mean_MW = recording.get_stat('mean', data_type='log_band_power', state='MW', filtered=filtered)

                if mean_OT is None or mean_MW is None:
                    print(f"Skipping recording for subject {subject.id} due to missing data.")
                    continue
                mean_diff = mean_OT - mean_MW
                data_records.append({
                    "dataset": dataset.name,
                    "subject": subject.id,
                    "group": subject.group,
                    "session": recording.session_id,
                    "task_orientation": dataset.task_orientation,
                    "mean_OT_alpha": mean_OT,
                    "mean_MW_alpha": mean_MW,
                    "mean_diff": mean_diff,
                    "filtered": filtered
                })
    return pd.DataFrame(data_records)


def run_and_plot_orientation_test(X, X_OT, X_MW, task_orientation, adjacency, info, output_dir, test_name):
    """
    Runs cluster permutation tests for a specific orientation (both tails) and plots the results.
    """
    results = {}
    tails = {'greater': 1, 'less': -1}
    mean_alpha_power_diff = X.mean(axis=0)
    mean_alpha_power_OT = X_OT.mean(axis=0)
    mean_alpha_power_MW = X_MW.mean(axis=0)

    for tail_name, tail_val in tails.items():
        T_obs, clusters, pvals, _ = perform_cluster_permutation_test(X, adjacency, tail=tail_val)
        mask, topomap_args, pval = get_significant_mask(clusters, pvals)

        if mask is not None:
            # Determine if test is confirmatory or exploratory
            if (task_orientation == 'internal' and tail_name == 'greater') or \
               (task_orientation == 'external' and tail_name == 'less'):
                test_type = "Confirmatory"
            else:
                test_type = "Exploratory"

            main_title = f"{test_type} Test for {task_orientation.capitalize()} Orientation (n_observations={X.shape[0]})"
            right_plot_title = f"t-values | Tail: {tail_name} | p-val={pval:.4f}"
            
            plot_task_orientation_map(mean_alpha_power_OT, mean_alpha_power_MW, mean_alpha_power_diff, T_obs, mask, info, main_title, right_plot_title, output_dir, tail_name, topomap_args)

        results[tail_name] = pvals
        print(f"Test: {test_name}, Orientation: {task_orientation.capitalize()}, Tail: {tail_name}, p-vals: {pvals}")
    return results


def run_analysis_and_save_results(analyzer, df, test_name, adjacency, info, ch_names):
    """
    Runs the full analysis pipeline for a given DataFrame and saves the results.
    """
    output_dir = os.path.join(analyzer.derivatives_path, "cluster_test_results", test_name)
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(os.path.join(output_dir, "recordings_and_data.csv"), index=False)

    X_internal = build_X(df, task_orientation='internal')
    X_external = build_X(df, task_orientation='external')
    
    X_internal_OT, X_internal_MW = build_state_vectors(df, task_orientation='internal')
    X_external_OT, X_external_MW = build_state_vectors(df, task_orientation='external')

    all_results = []
    if X_internal.size > 0:
        pvals_internal = run_and_plot_orientation_test(X_internal, X_internal_OT, X_internal_MW, 'internal', adjacency, info, output_dir, test_name)
        for tail, pvals in pvals_internal.items():
            for i, pval in enumerate(pvals):
                all_results.append({'task_orientation': 'internal', 'tail': tail, 'cluster_index': i, 'p_value': pval})

    if X_external.size > 0:
        pvals_external = run_and_plot_orientation_test(X_external, X_external_OT, X_external_MW, 'external', adjacency, info, output_dir, test_name)
        for tail, pvals in pvals_external.items():
            for i, pval in enumerate(pvals):
                all_results.append({'task_orientation': 'external', 'tail': tail, 'cluster_index': i, 'p_value': pval})

    pd.DataFrame(all_results).to_csv(os.path.join(output_dir, "cluster_test_p_values.csv"), index=False)

    mean_alpha_data = {"Channel Names": ch_names}
    if X_internal.size > 0:
        mean_alpha_data["Internal"] = X_internal.mean(axis=0)
    if X_external.size > 0:
        mean_alpha_data["External"] = X_external.mean(axis=0)
    pd.DataFrame(mean_alpha_data).to_csv(os.path.join(output_dir, "mean_alpha_power.csv"), index=False)


if __name__ == "__main__":
    ANALYZER_NAME = "eeg_analyzer_2"

    analyzer = EEGAnalyzer.load_analyzer(ANALYZER_NAME)
    if analyzer is None:
        print(f"Analyzer {ANALYZER_NAME} not found. Creating a new one.")
        analyzer = EEGAnalyzer(DATASETS, ANALYZER_NAME)
        analyzer.save_analyzer()

    ch_names = None
    for dataset in analyzer:
        for subject in dataset:
            ch_names = subject.get_channel_names()
            if ch_names:
                break
        if ch_names:
            break
    if not ch_names:
        raise ValueError("No channels found in any subject.")

    info = create_info(ch_names, sfreq=1, ch_types='eeg')
    info.set_montage(analyzer.get_montage())
    adjacency, _ = find_ch_adjacency(info, ch_type='eeg')

    # --- Test 1: No control subjects ---
    print("--- Running analysis with no control subjects ---")
    df_exp_only = extract_dataframe(analyzer, skip_controls=True, filter_task=False, filtered=True)
    run_analysis_and_save_results(analyzer, df_exp_only, "no_control_outlier_filtered", adjacency, info, ch_names)

    # --- Test 2: All subjects ---
    print("--- Running analysis for all subjects ---")
    df_all_subjects = extract_dataframe(analyzer, skip_controls=False, filter_task=False, filtered=True)
    run_analysis_and_save_results(analyzer, df_all_subjects, "all_subjects_outlier_filtered", adjacency, info, ch_names)

    # --- Test 3: No control, no filter ---
    print("--- Running analysis with no control subjects and no filtering ---")
    df_no_control_no_filter = extract_dataframe(analyzer, skip_controls=True, filter_task=False, filtered=False)
    run_analysis_and_save_results(analyzer, df_no_control_no_filter, "no_control_no_filter", adjacency, info, ch_names)

    # --- Test 4: All subjects, no filter ---
    print("--- Running analysis for all subjects with no filtering ---")
    df_all_no_filter = extract_dataframe(analyzer, skip_controls=False, filter_task=False, filtered=False)
    run_analysis_and_save_results(analyzer, df_all_no_filter, "all_subjects_no_filter", adjacency, info, ch_names)

    # --- Test 5: Filter task-specific data ---
    print("--- Running analysis with task-specific filtering (jin2019 dataset) ---")
    df_task_filtered = extract_dataframe(analyzer, skip_controls=False, filter_task=True, filtered=True)
    run_analysis_and_save_results(analyzer, df_task_filtered, "task_filtered_outlier_filtered", adjacency, info, ch_names)

    print("Cluster permutation analysis complete.")
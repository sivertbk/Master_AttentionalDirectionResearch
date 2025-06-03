import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import os
import pandas as pd
import mne
import seaborn as sns

from eeg_analyzer.dataset import Dataset
from utils.config import EEG_SETTINGS, PLOTS_PATH, DATASETS

band = (8, 12)

def plot_topomap_comparison(topo_data_ot: np.ndarray, topo_data_mw: np.ndarray, channels: list[str], band: tuple = (8, 12)) -> plt.Figure:
    """
    Plots a comparison of topographic maps for On-Target (OT) and Mind-Wandering (MW) states,
    as well as their difference.

    Args:
        ot_z_scores (np.ndarray, shape (n_channels,)): data for On-Target state.
        mw_z_scores (np.ndarray, shape (n_channels,)): data for Mind-Wandering state.
        channels (list): List of channel names corresponding to the topographic data.
        band (tuple): Frequency band of interest (e.g., (8, 12) for alpha band).
    Returns:
        fig (matplotlib.figure.Figure): Matplotlib figure object.
    """
    topo_data_diff = None
    if topo_data_ot is not None and topo_data_mw is not None:
        topo_data_diff = topo_data_ot - topo_data_mw

    montage_type = EEG_SETTINGS["MONTAGE"]
    info = mne.create_info(ch_names=channels, sfreq=128, ch_types="eeg")
    info.set_montage(montage_type)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plot_data_list = [topo_data_ot, topo_data_mw, topo_data_diff]
    plot_titles = ["On-task", "Mind-wandering", "Difference (OT-MW)"]

    # Determine vlim for OT and MW plots (shared scale)
    vlim_ot_mw = None
    valid_ot_mw_data = [d for d in [topo_data_ot, topo_data_mw] if d is not None]
    if valid_ot_mw_data:
        min_val = np.min([np.min(d) for d in valid_ot_mw_data])
        max_val = np.max([np.max(d) for d in valid_ot_mw_data])
        vlim_ot_mw = (min_val, max_val)
        if min_val == max_val: # Avoid vlim=(x,x) if data is flat
            vlim_ot_mw = (min_val - 0.03, max_val + 0.03) if min_val != 0 else (-0.1, 0.1)


    # Determine vlim for difference plot (symmetrical around 0)
    vlim_diff = None
    if topo_data_diff is not None:
        abs_max_diff = np.max(np.abs(topo_data_diff))
        if abs_max_diff == 0: # Data is all zeros
                vlim_diff = (-0.1, 0.1) # Small range for visual clarity
        else:
                vlim_diff = (-abs_max_diff, abs_max_diff)
    
    vlims_list = [vlim_diff, vlim_diff, vlim_diff]

    for i, ax in enumerate(axes):
        data = plot_data_list[i]
        title = plot_titles[i]
        current_vlim = vlims_list[i]
        
        # Determine cmap: explicit for difference, default for others
        current_cmap = None
        if title == "Difference (OT-MW)":
            current_cmap = 'coolwarm' # Explicitly set for the difference plot
        else:
            current_cmap = "coolwarm"

        if data is not None:
            im, _ = mne.viz.plot_topomap(
                data, 
                info, 
                axes=ax, 
                show=False, 
                contours=4, 
                vlim=current_vlim, 
                cmap=current_cmap  # Use the determined colormap
            )
            # Remove colorbar
            # cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            # if title == "Difference (OT-MW)":
            #     cbar.set_label(colorbar_label, rotation=270, labelpad=15)
            ax.axis('off')
            # ax.set_title(title, fontsize=12) # Remove title
        else:
            ax.axis('off')
            # ax.set_title(title, fontsize=12) # Remove title
            # ax.text(0.5, 0.5, "Data not available", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            # ax.axis('off')

    # Add minus and equals symbols
    fig.text(0.33333, 0.41, "$-$", ha='center', va='center', fontsize=40)
    fig.text(0.66667, 0.41, "$=$", ha='center', va='center', fontsize=40)

    main_title_parts = [
        f"Dataset: Braboszcz",
        f"Subject: All",
        f"Session: All",
        f"Band: {band[0]}–{band[1]} Hz"
    ]
    
    # fig.suptitle(" | ".join(main_title_parts), fontsize=16, y=0.98) # Remove supertitle
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle and bottom

    return fig


if __name__ == '__main__':
    dataset = Dataset(DATASETS["braboszcz2017"])

    dataset.load_subjects()
    exclude = dataset.get_group_subjects('ctr')
    dataset.remove_subjects(exclude)

    df = pd.DataFrame(dataset.to_long_band_power_list(band))
    
    df_z = df.groupby(['subject_session', 'channel'])['band_db'].transform(lambda x: (x - x.mean()) / x.std())
    df = df.assign(z_score=df_z)

    df['is_bad'] = np.abs(df['z_score']) > 3 # threshold for z-score to identify outliers

    print(f"Number of outliers detected: {df['is_bad'].sum()} out of {len(df)} total rows ({df['is_bad'].mean() * 100:.2f}%).")

    df = df[~df['is_bad']]  # Remove outliers based on z-score

    # make an array of z-score values for each channel and state and then get the average z-score for that channel and state
    df_z_avg = df.groupby(['channel', 'state'])['z_score'].mean().reset_index()


    # Get unique channel names
    channel_names = df['channel'].unique().tolist()

    # Create arrays for each state
    mw_z_scores = df_z_avg[df_z_avg['state'] == 'MW']['z_score'].values
    ot_z_scores = df_z_avg[df_z_avg['state'] == 'OT']['z_score'].values


    zero_scores = np.array([0] * 64)

    print("MW Z-scores:", mw_z_scores)
    print("OT Z-scores:", ot_z_scores)
    print("Channel Names:", channel_names)

    # Call the plotting function
    fig = plot_topomap_comparison(ot_z_scores, mw_z_scores, channel_names)

    # Save the figure to the specified path with no background
    fig_path = os.path.join(PLOTS_PATH, f"topomap_comparison_OG.png")
    fig.savefig(fig_path, format='png', bbox_inches='tight', transparent=True)



"""
Visualizer
----------

Responsible for generating plots and figures from EEG analysis results.

Responsibilities:
- Visualize statistical outcomes (e.g., topoplots, violin plots, effect size maps).
- Display raw or processed metrics for quality control and reporting.
- Maintain a consistent figure style across the project.

Notes:
- Should not compute statistics or metrics; only display them.
"""


import numpy as np
import mne
import mne.viz as viz
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os 

from utils.config import set_plot_style, channel_positions, PLOTS_PATH, EEG_SETTINGS

set_plot_style()

class Visualizer:
    """
    Class containing methods to visualize EEG PSD data.
    """
    def __init__(self, analyzer):
        """
        Initialize the Visualizer.
        """
        self.analyzer = analyzer
        self.analyzer_name = analyzer.analyzer_name
        self.derivatives_path = analyzer.derivatives_path

    def plot_boxplot(self, 
                     value_col: str = 'log_band_power', 
                     state_col: str = 'state', 
                     output_subfolder: str = 'boxplots',
                     exclude_bads: bool = True,
                     group_by_region: bool = False):
        """
        Generates and saves boxplots for each channel, separated by state, 
        for each subject_session within each dataset.

        Plots are saved to: derivatives_path/output_subfolder/dataset_name/

        Parameters:
        - value_col (str): The column name for the y-axis values (e.g., 'band_db').
        - state_col (str): The column name for the state (e.g., 'state'), expected to contain 0 and 1.
        - output_subfolder (str): Name of the subfolder within the analyzer's derivatives_path 
                                  where plots will be saved.
        - exclude_bads (bool): If True, excludes rows where 'is_bad' is True from the plots.
        - group_by_region (bool): If True, groups boxplots by cortical region instead of channel.
        """

        df = self.analyzer.df
        if df is None or df.empty:
            print(f"[Visualizer for {self.analyzer_name}] DataFrame is empty. Cannot generate boxplots.")
            return

        grouping_col = 'cortical_region' if group_by_region else 'channel'
        required_cols = ['dataset', 'subject_session', state_col, value_col, grouping_col]
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"[Visualizer for {self.analyzer_name}] DataFrame is missing required columns: {missing_cols}. Cannot generate boxplots.")
            return
        

        # Fix derivatives path if it's a Unix-style path on Windows
        derivatives_path = self.derivatives_path
        
        plots_main_dir = os.path.join(derivatives_path, output_subfolder, 'channel_boxplots' if not group_by_region else 'region_boxplots')
        # Normalize path for current OS
        plots_main_dir = os.path.abspath(plots_main_dir)
        os.makedirs(plots_main_dir, exist_ok=True)
        
        print(f"[Visualizer for {self.analyzer_name}] Base derivatives path: {self.derivatives_path}")
        print(f"[Visualizer for {self.analyzer_name}] Corrected derivatives path: {derivatives_path}")
        print(f"[Visualizer for {self.analyzer_name}] Main plots directory: {plots_main_dir}")
        print(f"[Visualizer for {self.analyzer_name}] Current working directory: {os.getcwd()}")
        print(f"[Visualizer for {self.analyzer_name}] Platform: {os.name}")
        print(f"[Visualizer for {self.analyzer_name}] Path separator: {os.sep}")

        # User-provided labels for the legend
        state_legend_labels = {"OT": "On-target", "MW": "Mind-wandering"} 
        
        cmap = sns.color_palette("coolwarm", as_cmap=True)
        # Palette for 'OT' and 'MW' states after mapping
        plot_palette = {"OT": cmap(0.1), "MW": cmap(0.9)} # Adjusted for distinct cool/warm

        for dataset_name in df['dataset'].unique():
            # Sanitize dataset name for use in file paths
            safe_dataset_name = str(dataset_name).replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
            dataset_plot_dir = os.path.join(plots_main_dir, safe_dataset_name)
            # Normalize path for current OS
            dataset_plot_dir = os.path.abspath(dataset_plot_dir)
            os.makedirs(dataset_plot_dir, exist_ok=True)
            
            df_dataset = df[df['dataset'] == dataset_name]
            
            for subject_session_id in df_dataset['subject_session'].unique():
                group_df = df_dataset[df_dataset['subject_session'] == subject_session_id].copy() # Use .copy() to avoid SettingWithCopyWarning

                if group_df.empty:
                    continue

                unique_groups = group_df[grouping_col].unique()

                if group_by_region:
                    # Define the desired order of cortical regions
                    region_order = ['prefrontal', 'frontal', 'frontocentral', 'central', 
                                    'centroparietal', 'temporal', 'parietal', 'parietooccipital', 'occipital']
                    # Filter out regions not in the defined order and then sort
                    sorted_groups = [region for region in region_order if region in unique_groups]
                else:
                    # Sort channels by Y-coordinate (anterior to posterior)
                    sorted_groups = sorted(
                        unique_groups,
                        key=lambda ch: channel_positions.get(ch, (0, float('inf'), 0))[1] # Sort unknown channels last
                    )

                if not sorted_groups:
                    continue

                fig, ax = plt.subplots(figsize=(max(15, len(sorted_groups) * 0.8), 8))
                
                sns.boxplot(x=grouping_col, y=value_col, hue='state', data=group_df, 
                            order=sorted_groups, hue_order=["OT", "MW"], palette=plot_palette, ax=ax)

                title = f'Band Power (8 - 12 Hz) by {grouping_col.replace("_", " ").title()} and State\nDataset: {dataset_name} - Subject Session: {subject_session_id}'
                ax.set_title(title, fontsize=16)
                xlabel = grouping_col.replace("_", " ").title()
                ax.set_xlabel(xlabel, fontsize=14)
                ax.set_ylabel(value_col.replace("_", " ").title(), fontsize=14)
                
                ax.tick_params(axis='x', labelrotation=45)
                for label in ax.get_xticklabels():
                    label.set_ha('right') # Align rotated labels

                # Customize legend
                handles, _ = ax.get_legend_handles_labels()
                ax.legend(handles, [state_legend_labels["OT"], state_legend_labels["MW"]], title=state_col.capitalize(), title_fontsize='13', fontsize='12')

                # Add text under x-axis, centered
                if not group_by_region:
                    ax.annotate(
                        "Posterior <------------------------> Anterior", # Adjusted arrow direction based on typical EEG layouts where anterior is often top/smaller Y
                        xy=(0.5, -0.22), # Adjusted y position for potentially rotated labels
                        xycoords='axes fraction',
                        ha='center',
                        va='center',
                        fontsize=12
                    )
                
                plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust rect to make space for annotation and title

                fig_filename = f"{safe_dataset_name}_{subject_session_id}_{value_col}_{grouping_col}_state_boxplot.svg"
                fig_path = os.path.join(dataset_plot_dir, fig_filename)
                # Normalize path for current OS
                fig_path = os.path.abspath(fig_path)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(fig_path), exist_ok=True)
                
                try:
                    plt.savefig(fig_path, format='svg', bbox_inches='tight')
                    print(f"[Visualizer for {self.analyzer_name}] Saved plot: {fig_path}")
                except Exception as e:
                    print(f"[Visualizer for {self.analyzer_name}] Error saving plot {fig_path}: {e}")
                    # Additional debugging info
                    print(f"[Visualizer for {self.analyzer_name}] Directory exists: {os.path.exists(os.path.dirname(fig_path))}")
                    print(f"[Visualizer for {self.analyzer_name}] Directory path: {os.path.dirname(fig_path)}")
                    print(f"[Visualizer for {self.analyzer_name}] File path absolute: {os.path.abspath(fig_path)}")
                    print(f"[Visualizer for {self.analyzer_name}] Current working directory: {os.getcwd()}")
                
                plt.close(fig)
        
        print(f"[Visualizer for {self.analyzer_name}] Finished generating boxplots. Saved to: {plots_main_dir}")


    @staticmethod
    def plot_topomap_comparison(topo_data_ot: np.ndarray, 
                                topo_data_mw: np.ndarray, 
                                channels: list[str], 
                                significance: list[bool] = None, 
                                band: tuple = (8, 12)) -> plt.Figure:
        """
        Plots a comparison of topographic maps for On-Task (OT) and Mind-Wandering (MW) states,
        as well as their difference.

        Args:
            topo_data_ot (np.ndarray, shape (n_channels,)): data for On-Task state.
            topo_data_mw (np.ndarray, shape (n_channels,)): data for Mind-Wandering state.
            channels (list): List of channel names corresponding to the topographic data.
            significance (list[bool], optional): Significance values for each channel, used to highlight significant channels.
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

        # If significance is provided, set up mask
        if significance is not None:
            significance = np.array(significance, dtype=bool)
        
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

        colorbar_label = "z-score"

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
                    mask=significance if significance is not None else None,  # Apply mask if provided
                    mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=5) if significance is not None else None,
                    vlim=current_vlim,
                    cmap=current_cmap  # Use the determined colormap
                )
                cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
                if title == "Difference (OT-MW)":
                    cbar.set_label(colorbar_label, rotation=270, labelpad=15)
                ax.set_title(title, fontsize=12)
            else:
                ax.set_title(title, fontsize=12)
                ax.text(0.5, 0.5, "Data not available", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.axis('off')

        main_title_parts = [
            f"Dataset: Braboszcz",
            f"Subject: All",
            f"Session: All",
            f"Band: {band[0]}–{band[1]} Hz"
        ]
        
        fig.suptitle(" | ".join(main_title_parts), fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle and bottom

        return fig
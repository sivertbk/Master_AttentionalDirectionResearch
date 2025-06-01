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
import os # Added for path operations

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
                     value_col: str = 'band_db', 
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
        required_cols = ['dataset', 'subject_session', 'is_bad', state_col, value_col, grouping_col]
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"[Visualizer for {self.analyzer_name}] DataFrame is missing required columns: {missing_cols}. Cannot generate boxplots.")
            return
        
        # Filter out bad data if exclude_bads is True
        if exclude_bads:
            df = df[df['is_bad'] == False]

        plots_main_dir = os.path.join(self.derivatives_path, output_subfolder, 'channel_boxplots' if not group_by_region else 'region_boxplots')
        os.makedirs(plots_main_dir, exist_ok=True)

        # User-provided labels for the legend
        state_legend_labels = {"OT": "On-target", "MW": "Mind-wandering"} 
        
        cmap = sns.color_palette("coolwarm", as_cmap=True)
        # Palette for 'OT' and 'MW' states after mapping
        plot_palette = {"OT": cmap(0.1), "MW": cmap(0.9)} # Adjusted for distinct cool/warm

        for dataset_name in df['dataset'].unique():
            dataset_plot_dir = os.path.join(plots_main_dir, str(dataset_name))
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

                title = f'Band Power ({self.analyzer.freq_band[0]} - {self.analyzer.freq_band[1]} Hz) by {grouping_col.replace("_", " ").title()} and State\nDataset: {dataset_name} - Subject Session: {subject_session_id}'
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

                fig_filename = f"{dataset_name}_{subject_session_id}_{value_col}_{grouping_col}_state_boxplot.svg"
                fig_path = os.path.join(dataset_plot_dir, fig_filename)
                
                try:
                    plt.savefig(fig_path, format='svg', bbox_inches='tight')
                    print(f"[Visualizer for {self.analyzer_name}] Saved plot: {fig_path}")
                except Exception as e:
                    print(f"[Visualizer for {self.analyzer_name}] Error saving plot {fig_path}: {e}")
                
                plt.close(fig)
        
        print(f"[Visualizer for {self.analyzer_name}] Finished generating boxplots. Saved to: {plots_main_dir}")



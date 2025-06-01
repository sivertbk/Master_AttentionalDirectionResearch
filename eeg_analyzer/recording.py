"""
Recording
---------

Represents a single EEG recording session for a subject.

Responsibilities:
- Store preprocessed PSD data (e.g., per task and state).
- Provide methods to access PSDs or derived features like alpha power.
- Manage loading/saving of EEG data at the session level.

Notes:
- Even if a subject has only one recording, this abstraction keeps the structure consistent.
- Can hold references to metadata like task labels or state mappings.
"""

from collections import defaultdict
import numpy as np
import os
import seaborn as sns
import pandas as pd

from eeg_analyzer.metrics import Metrics
from utils.config import EEG_SETTINGS, set_plot_style, PLOTS_PATH
import mne
import matplotlib.pyplot as plt

set_plot_style()  # Set the plotting style for MNE and Matplotlib
class Recording:
    def __init__(self, session_id: int, psd_entries: list[np.ndarray], metadata_entries: list[dict], freq_entries: list[np.ndarray], channels: list[str]):
        self.session_id = session_id
        self.dataset_name = None 
        self.dataset_f_name = None 
        self.subject_id = None  
        self.psd_map = defaultdict(dict)     # task -> state -> PSD
        self.meta_map = defaultdict(dict)    # task -> state -> metadata
        self.freq_map = defaultdict(dict)    # task -> state -> frequencies
        self.channels = channels             # List of channel names

        for psd, meta, freqs in zip(psd_entries, metadata_entries, freq_entries):
            task = meta["task"]
            state = meta["state"]
            self.psd_map[task][state] = psd
            self.meta_map[task][state] = meta
            self.freq_map[task][state] = freqs
            if self.dataset_name is None:
                self.dataset_name = meta.get("dataset_name", "Unknown Dataset")
            if self.dataset_f_name is None:
                self.dataset_f_name = meta.get("dataset_f_name", "Unknown Dataset Filename")
            if self.subject_id is None:
                self.subject_id = meta.get("subject", "Unknown Subject")

    def __repr__(self):
        total_conditions = sum(len(states) for states in self.psd_map.values())
        return f"<Recording session-{self.session_id} with {total_conditions} condition(s)>"
    
    #                                           Public API
    ##########################################################################################################
    
    def get_psd(self, task: str, state: str):
        try:
            return self.psd_map[task][state]
        except KeyError:
            raise ValueError(f"No PSD found for task '{task}' and state '{state}' in session {self.session_id}")

    def get_freqs(self, task: str, state: str) -> np.ndarray:
        try:
            return self.freq_map[task][state]
        except KeyError:
            raise ValueError(f"No frequency data for task '{task}' and state '{state}' in session {self.session_id}")

    def get_metadata(self, task: str, state: str):
        try:
            return self.meta_map[task][state]
        except KeyError:
            raise ValueError(f"No metadata found for task '{task}' and state '{state}' in session {self.session_id}")
        
    def get_channel_names(self):
        """Return the list of channel names."""
        return self.channels

    def get_available_tasks(self):
        return list(self.psd_map.keys())

    def get_available_states(self, task: str):
        return list(self.psd_map[task].keys()) if task in self.psd_map else []
    
    def get_num_epochs(self):
        """Return a dictionary with number of epochs for each task and state."""
        num_epochs = {}
        for task, states in self.psd_map.items():
            for state, psd in states.items():
                num_epochs[(task, state)] = psd.shape[0]
        return num_epochs

    def list_conditions(self) -> list[tuple[str, str]]:
        """List all available (task, state) condition pairs."""
        return [
            (task, state)
            for task, states in self.psd_map.items()
            for state in states
        ]

    def get_mw_ot_epoch_ratio(self) -> float:
        """
        Calculate the ratio of total MW epochs to total OT epochs across all tasks.
        Returns np.inf if OT epochs count is zero.
        """
        total_mw_epochs = 0
        total_ot_epochs = 0

        for task, states_data in self.psd_map.items():
            for state, psd_array in states_data.items():
                num_epochs_for_condition = psd_array.shape[0]
                if state == "MW":
                    total_mw_epochs += num_epochs_for_condition
                elif state == "OT":
                    total_ot_epochs += num_epochs_for_condition
        
        if total_ot_epochs == 0:
            return np.inf  # Avoid division by zero
        
        return total_mw_epochs / total_ot_epochs
    
    def alpha_power(self, task: str, state: str) -> np.ndarray:
        """Compute alpha power for a given task and state."""
        psd = self.get_psd(task, state)
        freqs = self.get_freqs(task, state)
        return Metrics.alpha_power(psd, freqs)
    
    def mean_alpha_power_per_channel(self, task: str, state: str) -> np.ndarray:
        """Compute mean alpha power per channel for a given task and state."""
        alpha_power = self.alpha_power(task, state)
        return alpha_power.mean(axis=0)
    
    def stats_band_power_per_channel(self, task: str, state: str, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute mean, variance, and standard error of band power for a given task and state.

        Parameters:
        - task: Task name (e.g., 'task1').
        - state: State name (e.g., 'MW', 'OT').
        - band: Frequency band as a tuple (low_freq, high_freq) in Hz.

        Returns:
        - A tuple containing:
            - Mean band power per channel.
            - Variance of band power per channel.
            - Standard error of band power per channel.
        """
        psd = self.get_psd(task, state)
        freqs = self.get_freqs(task, state)
        band_power = Metrics.band_power(psd, freqs, band)
        mean = band_power.mean(axis=0)
        var = band_power.var(axis=0)
        std_err = band_power.std(axis=0) / np.sqrt(band_power.shape[0])
        return mean, var, std_err


    def plot_topo_power(self, condition: tuple[str, str], band: tuple[float, float] = (8, 12), show: bool = True) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot topographic power map for a given task and state or for OT–MW difference.

        Parameters:
        - condition: tuple (task, state), where state can be 'MW', 'OT', or 'difference'.
                    If task is 'all', averages across all tasks in the recording.
        - band: tuple (low_freq, high_freq) specifying the frequency band in Hz.
        """
        task_input, state = condition
        montage_type = EEG_SETTINGS["MONTAGE"]

        # Get all available conditions
        available_conditions = self.list_conditions()

        # Determine which task–state pairs to use
        if task_input == "all":
            tasks = set(t for (t, s) in available_conditions)
        else:
            tasks = [task_input]

        # Prepare to store power values
        power_arrays = []

        for task in tasks:
            if state == "difference":
                available_tasks = self.get_available_tasks()

                mw_tasks = [t for t in available_tasks if "MW" in self.get_available_states(t)]
                ot_tasks = [t for t in available_tasks if "OT" in self.get_available_states(t)]

                mw_arrays, ot_arrays = [], []

                for t in mw_tasks:
                    alpha = self.stats_band_power_per_channel(t, "MW", band)[0]
                    mw_arrays.append(alpha)

                for t in ot_tasks:
                    alpha = self.stats_band_power_per_channel(t, "OT", band)[0]
                    ot_arrays.append(alpha)

                if not mw_arrays or not ot_arrays:
                    print("Insufficient data for computing OT–MW difference.")
                    return

                alpha_mw = np.mean(mw_arrays, axis=0)
                alpha_ot = np.mean(ot_arrays, axis=0)

                power_arrays.append(alpha_ot - alpha_mw)
            else:
                if (task, state) not in available_conditions:
                    continue
                psd = self.get_psd(task, state)
                freqs = self.get_freqs(task, state)
                alpha = Metrics.band_power(psd, freqs, band, operation='mean').mean(axis=0)
                power_arrays.append(alpha)

        if not power_arrays:
            print("No valid data available for the requested condition.")
            return

        # Average across tasks if multiple arrays
        topo_data = np.mean(power_arrays, axis=0)

        # Create MNE info with channel locations
        info = mne.create_info(ch_names=self.channels, sfreq=128, ch_types="eeg")
        info.set_montage(montage_type)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(6, 5))

        # Plot and get the image object
        im, _ = mne.viz.plot_topomap(
            topo_data, info, axes=ax, show=False, contours=4
        )

        # Add vertical colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

        # Unit for colorbar
        unit = "µV²" 

        cbar.set_label(unit, rotation=270, labelpad=15)

        # Title and layout
        title = f"State: {state} | Band: {band[0]}–{band[1]} Hz"
        ax.set_title(title, fontsize=12)
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_topo_power_comparison(self, task_input: str, band: tuple[float, float] = (8, 12), show: bool = True, use_decibel: bool = False):
        """
        Plot topographic power maps for OT, MW, and OT-MW difference in one figure.

        Parameters:
        - task_input: str, task name or 'all' to average across all tasks.
        - band: tuple (low_freq, high_freq), frequency band in Hz.
        - show: bool, whether to display the plot.
        - use_decibel: bool, whether to plot power in decibels.
        """
        montage_type = EEG_SETTINGS["MONTAGE"]
        info = mne.create_info(ch_names=self.channels, sfreq=128, ch_types="eeg")
        info.set_montage(montage_type)

        available_conditions = self.list_conditions()
        if task_input == "all":
            tasks_to_process = sorted(list(set(t for (t, s) in available_conditions)))
        else:
            tasks_to_process = [task_input]

        # Helper function to get averaged power for a given state
        def get_averaged_power_for_state(state_str: str, tasks_list: list[str], band_tuple: tuple[float, float]) -> np.ndarray | None:
            power_arrays_for_state = []
            for task_iter in tasks_list:
                if (task_iter, state_str) in available_conditions:
                    try:
                        psd_val = self.get_psd(task_iter, state_str)
                        freqs_val = self.get_freqs(task_iter, state_str)
                        if use_decibel:
                            power_val = Metrics.band_decibel(psd_val, freqs_val, band_tuple, operation='mean').mean(axis=0)
                        else:
                            power_val = Metrics.band_power(psd_val, freqs_val, band_tuple, operation='mean').mean(axis=0)
                        power_arrays_for_state.append(power_val)
                    except ValueError:
                        # This might happen if data is corrupted or unexpectedly missing
                        print(f"Warning: Could not retrieve data for task '{task_iter}', state '{state_str}'. Skipping.")
                        continue
            
            if not power_arrays_for_state:
                return None
            return np.mean(power_arrays_for_state, axis=0)

        topo_data_ot = get_averaged_power_for_state("OT", tasks_to_process, band)
        topo_data_mw = get_averaged_power_for_state("MW", tasks_to_process, band)

        topo_data_diff = None
        if topo_data_ot is not None and topo_data_mw is not None:
            topo_data_diff = topo_data_ot - topo_data_mw
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        plot_data_list = [topo_data_ot, topo_data_mw, topo_data_diff]
        plot_titles = ["On-target", "Mind-wandering", "Difference (OT-MW)"]

        # Determine vlim for OT and MW plots (shared scale)
        vlim_ot_mw = None
        valid_ot_mw_data = [d for d in [topo_data_ot, topo_data_mw] if d is not None]
        if valid_ot_mw_data:
            min_val = np.min([np.min(d) for d in valid_ot_mw_data])
            max_val = np.max([np.max(d) for d in valid_ot_mw_data])
            vlim_ot_mw = (min_val, max_val)
            if min_val == max_val: # Avoid vlim=(x,x) if data is flat
                vlim_ot_mw = (min_val - 0.1, max_val + 0.1) if min_val != 0 else (-0.1, 0.1)


        # Determine vlim for difference plot (symmetrical around 0)
        vlim_diff = None
        if topo_data_diff is not None:
            abs_max_diff = np.max(np.abs(topo_data_diff))
            if abs_max_diff == 0: # Data is all zeros
                 vlim_diff = (-0.1, 0.1) # Small range for visual clarity
            else:
                 vlim_diff = (-abs_max_diff, abs_max_diff)
        
        vlims_list = [vlim_ot_mw, vlim_ot_mw, vlim_diff]

        colorbar_label = "dB" if use_decibel else "µV²"

        for i, ax in enumerate(axes):
            data = plot_data_list[i]
            title = plot_titles[i]
            current_vlim = vlims_list[i]
            
            # Determine cmap: explicit for difference, default for others
            current_cmap = None
            if title == "Difference (OT-MW)":
                current_cmap = 'coolwarm' # Explicitly set for the difference plot
            else:
                current_cmap = "Reds"

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
                cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
                cbar.set_label(colorbar_label, rotation=270, labelpad=15)
                ax.set_title(title, fontsize=12)
            else:
                ax.set_title(title, fontsize=12)
                ax.text(0.5, 0.5, "Data not available", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.axis('off')

        main_title_parts = [
            f"Dataset: {self.dataset_name}",
            f"Subject: {self.subject_id}",
            f"Session: {self.session_id}",
            f"Band: {band[0]}–{band[1]} Hz"
        ]
        if use_decibel:
            main_title_parts.append("Scale: dB")
        if task_input != "all":
            main_title_parts.append(f"Task: {task_input}")
        
        fig.suptitle(" | ".join(main_title_parts), fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle and bottom

        # Save figure to Plots path
        if use_decibel:
            save_dir = os.path.join(PLOTS_PATH, self.dataset_f_name, "topo_power_comparison", f"band-{band[0]}-{band[1]}", "decibel")
        else:
            save_dir = os.path.join(PLOTS_PATH, self.dataset_f_name, "topo_power_comparison", f"band-{band[0]}-{band[1]}")
        os.makedirs(save_dir, exist_ok=True)
        decibel_suffix = "_db" if use_decibel else ""
        file_name = f"topo_power_comparison_sub-{self.subject_id}_ses-{self.session_id}_task-{task_input}_band-{band[0]}-{band[1]}{decibel_suffix}.svg"
        file_path = os.path.join(save_dir, file_name)
        fig.savefig(file_path, format='svg', bbox_inches='tight')
        print(f"Topographic power comparison saved to {file_path}")
        
        if show:
            plt.show()
        plt.close()
        
        return fig, axes

    def plot_psd(self, show: bool = True):
        """
        Plot the Power Spectral Density (PSD) in decibel for all task-state pairs.
        The area under each line is filled.
        """
        plt.figure(figsize=(12, 7))
        
        conditions = self.list_conditions()
        if not conditions:
            print(f"No conditions to plot for subject {self.subject_id}, session {self.session_id}.")
            plt.close() # Close the empty figure
            return

        colors = plt.cm.get_cmap('tab10', len(conditions))
        
        all_mean_psd_db = []
        all_freqs = []

        for i, (current_task, current_state) in enumerate(conditions):
            psd = self.get_psd(current_task, current_state)
            freqs = self.get_freqs(current_task, current_state)
            psd_db = Metrics.to_db(psd)
            mean_psd_db = psd_db.mean(axis=(0, 1))  # Average across epochs and channels
            all_mean_psd_db.append(mean_psd_db)
            all_freqs.append(freqs)

        # Determine a global minimum for the y-axis fill baseline
        # This ensures fill is always "under" the line, even for negative values
        global_min_val = np.min([np.min(psd_data) for psd_data in all_mean_psd_db if psd_data.size > 0])
        # Add a small padding below the global minimum for the fill
        fill_baseline = global_min_val - (np.abs(global_min_val) * 0.01 if global_min_val != 0 else 0.1)


        for i, (current_task, current_state) in enumerate(conditions):
            mean_psd_db = all_mean_psd_db[i]
            freqs = all_freqs[i]
            line_color = colors(i)
            
            plt.plot(freqs, mean_psd_db, label=f"{current_task} - {current_state}", color=line_color)
            plt.fill_between(freqs, mean_psd_db, y2=fill_baseline, alpha=0.3, color=line_color, interpolate=True)

        plt.title(f"Dataset: {self.dataset_name} | Subject: {self.subject_id} | Session: {self.session_id} | All Task-State PSDs")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (dB)")
        plt.legend()
        plt.grid(True)

        # save the figure to Plots path
        save_dir = os.path.join(PLOTS_PATH, self.dataset_f_name, "psd_plots")
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"psd_plot_sub-{self.subject_id}_ses-{self.session_id}.svg"
        file_path = os.path.join(save_dir, file_name)
        plt.savefig(file_path, format='svg', bbox_inches='tight')
        print(f"PSD plot saved to {file_path}")

        if show:
            plt.show()
        
        plt.close()

    def plot_distribution(self, band: tuple[float, float] = (8, 12), show: bool = True, use_decibel: bool = True):
        """
        Plot the distribution of alpha power for each channel in the specified frequency band.
        Produces a histogram for each channel showing the distribution of alpha power with KDE.
        The histogram does not separate tasks or states, but rather shows the overall distribution
        of alpha power across all epochs.
        If use_decibel is True, uses decibel-transformed band power.
        """
        channel_names = self.get_channel_names()
        if not channel_names:
            print(f"No channels available for subject {self.subject_id}, session {self.session_id}.")
            return
        
        # Prepare data for Seaborn
        data = []
        for task, states in self.psd_map.items():
            for state, psd in states.items():
                freqs = self.freq_map[task][state]
                if use_decibel:
                    band_power = Metrics.band_decibel(psd, freqs, band, operation='mean')
                else:
                    band_power = Metrics.band_power(psd, freqs, band, operation='mean')
                # band_power shape: (epochs, channels)
                for epoch_idx in range(band_power.shape[0]):
                    for ch_idx, ch_name in enumerate(channel_names):
                        data.append({
                            'band_power': band_power[epoch_idx, ch_idx],
                            'condition': f"{task} - {state}",
                            'channel': ch_name
                        })

        if not data:
            print(f"No data available for plotting distribution for subject {self.subject_id}, session {self.session_id}.")
            return

        df = pd.DataFrame(data)

        # Plotting
        for channel_name in channel_names:
            plt.figure(figsize=(10, 6))
            
            channel_data = df[df['channel'] == channel_name]
            
            sns.histplot(
                data=channel_data,
                x='band_power',
                kde=True,
                stat='density',
                alpha=0.6,
                common_norm=False
            )
            
            plt.title(f'Band Power Distribution for Channel {channel_name} (Subject {self.subject_id})')
            plt.xlabel('Band Power (dB)' if use_decibel else 'Band Power (µV²/Hz)')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save the figure to Plots path
            save_dir = os.path.join(PLOTS_PATH, self.dataset_f_name, "distribution_plots", f"band-{band[0]}-{band[1]}")
            os.makedirs(save_dir, exist_ok=True)
            decibel_suffix = "_db" if use_decibel else ""
            file_name = f"distribution_plot_sub-{self.subject_id}_ses-{self.session_id}_channel-{channel_name}{decibel_suffix}.svg"
            file_path = os.path.join(save_dir, file_name)
            plt.savefig(file_path, format='svg', bbox_inches='tight')
            print(f"Distribution plot saved to {file_path}")

            if show:
                plt.show()
            
            plt.close()








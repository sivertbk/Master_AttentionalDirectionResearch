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

from eeg_analyzer.metrics import Metrics
from utils.config import EEG_SETTINGS, ROIs, channel_positions, cortical_regions
import mne
import matplotlib.pyplot as plt

class Recording:
    def __init__(self, session_id: int, psd_entries: list[np.ndarray], metadata_entries: list[dict], freq_entries: list[np.ndarray], channels: list[str]):
        self.session_id = session_id
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


    def plot_topo_power(self, condition: tuple[str, str], band: tuple[float, float] = (8, 12)):
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
                alpha = Metrics.band_power(psd, freqs, band).mean(axis=0)
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
            topo_data, info, axes=ax, show=False, contours=0
        )

        # Add vertical colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

        # Try to get the unit from any task/state combo that was actually used
        unit = "µV²/Hz"  # fallback
        try:
            if state != "difference":
                meta = self.get_metadata(next(iter(tasks)), state)
            else:
                meta = self.get_metadata(
                    next(iter(ot_tasks)) if ot_tasks else next(iter(mw_tasks)),
                    "OT" if ot_tasks else "MW"
                )
            unit = meta.get("psd_unit", "µV²/Hz").encode('utf-8').decode('unicode_escape')
        except Exception:
            pass

        cbar.set_label(unit, rotation=270, labelpad=15)

        # Title and layout
        title = f"Topomap: {task_input} | {state} | {band[0]}–{band[1]} Hz"
        ax.set_title(title, fontsize=12)
        plt.tight_layout()
        plt.show()




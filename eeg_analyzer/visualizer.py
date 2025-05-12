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
- Designed to be modular and compatible with Jupyter or script-based workflows.
"""


import numpy as np
import mne
import mne.viz as viz
import matplotlib.pyplot as plt

class Visualizer:
    """
    Class containing methods to visualize EEG PSD data.
    """

    import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    """
    Class containing visualization methods for EEG PSD data.
    """
    @staticmethod
    def plot_band_power(band_powers, channels, band_name="Frequency Band", figsize=(10, 5)):
        """
        Plot average power in a specific frequency band for each channel.

        Parameters:
            band_powers (ndarray): Band power array (epochs × channels).
            channels (list[str]): Channel names.
            band_name (str): Name of the frequency band.
            figsize (tuple): Figure size.
        """
        mean_band_power = np.mean(band_powers, axis=0)

        plt.figure(figsize=figsize)
        plt.bar(channels, mean_band_power, color='skyblue')
        plt.title(f'Average Power in {band_name.capitalize()} Band')
        plt.xlabel('Channels')
        plt.ylabel('Mean Power')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        plt.show()

    @staticmethod
    def compare_states(freqs, psd, metadata, channels, state_col='state', figsize=(10, 5)):
        """
        Compare PSD between two states (e.g., focused vs mind-wandering).

        Parameters:
            freqs (ndarray): Frequencies array.
            psd (ndarray): PSD array (epochs × channels × frequencies).
            metadata (DataFrame): Metadata DataFrame with state information.
            channels (list[str]): Channel names.
            state_col (str): Column name in metadata indicating the state.
            figsize (tuple): Figure size.
        """
        states = metadata[state_col].unique()
        if len(states) != 2:
            raise ValueError("This method currently supports exactly two states for comparison.")

        plt.figure(figsize=figsize)

        for state in states:
            state_indices = metadata[metadata[state_col] == state].index
            mean_psd_state = psd[state_indices].mean(axis=(0, 1))
            plt.plot(freqs, mean_psd_state, label=state.capitalize())

        plt.title(f'PSD Comparison: {states[0]} vs. {states[1]}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Mean Power')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_alpha_power_by_state(alpha_power: dict, states: list, channels: list, figsize=(12, 6)):
        """
        Plot alpha power for each state across channels as bar plots.

        Parameters:
            alpha_power (dict): Dictionary with states as keys and alpha power arrays (epochs × channels) as values.
            states (list): List of state names.
            channels (list): List of channel names.
            figsize (tuple): Figure size.
        """
        plt.figure(figsize=figsize)
        bar_width = 0.35
        x = np.arange(len(channels))

        for i, state in enumerate(states):
            mean_alpha_power = np.mean(alpha_power[state], axis=0)
            plt.bar(x + i * bar_width, mean_alpha_power, bar_width, label=f"State: {state}")

        plt.title("Alpha Power by State")
        plt.xlabel("Channels")
        plt.ylabel("Mean Alpha Power")
        plt.xticks(x + bar_width * (len(states) - 1) / 2, channels, rotation=90)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_alpha_power_by_task(alpha_power: dict, tasks: list, channels: list, figsize=(12, 6)):
        """
        Plot alpha power for each task across channels as bar plots.

        Parameters:
            alpha_power (dict): Dictionary with tasks as keys and alpha power arrays (epochs × channels) as values.
            tasks (list): List of task names.
            channels (list): List of channel names.
            figsize (tuple): Figure size.
        """
        plt.figure(figsize=figsize)
        bar_width = 0.35
        x = np.arange(len(channels))

        for i, task in enumerate(tasks):
            mean_alpha_power = np.mean(alpha_power[task], axis=0)
            plt.bar(x + i * bar_width, mean_alpha_power, bar_width, label=f"Task: {task}")

        plt.title("Alpha Power by Task")
        plt.xlabel("Channels")
        plt.ylabel("Mean Alpha Power")
        plt.xticks(x + bar_width * (len(tasks) - 1) / 2, channels, rotation=90)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

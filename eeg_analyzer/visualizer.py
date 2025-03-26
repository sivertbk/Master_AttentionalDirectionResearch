import numpy as np
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

    def __init__(self, subject):
        """
        Initialize Visualizer with a reference to a Subject instance.

        Parameters:
            subject (Subject): An instance of the Subject class.
        """
        self.subject = subject

    def plot_psd(self, epoch_idx=-1, channels='all', figsize=(10, 5)):
        """
        Plot PSD for a specific epoch or average across epochs.

        Parameters:
            epoch_idx (int): Index of epoch to plot, or -1 for average across epochs.
            channels (str or list[str]): 'all' or specific channel name(s).
            figsize (tuple): Figure size.
        """
        psd = self.subject.psd
        freqs = self.subject.freqs
        ch_names = self.subject.channels

        # Handle channel selection explicitly
        if channels == 'all':
            channel_indices = range(len(ch_names))
        else:
            if isinstance(channels, str):
                channels = [channels]
            channel_indices = [ch_names.index(ch) for ch in channels]

        # Handle epoch selection explicitly
        if epoch_idx == -1:
            psd_to_plot = psd[:, channel_indices, :].mean(axis=0)
            title = "Mean PSD across epochs"
        else:
            psd_to_plot = psd[epoch_idx, channel_indices, :]
            title = f"PSD for Epoch {epoch_idx}"

        plt.figure(figsize=figsize)
        for idx, ch_idx in enumerate(channel_indices):
            plt.plot(freqs, psd_to_plot[idx, :], label=ch_names[ch_idx])

        plt.title(title)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()



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

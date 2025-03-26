import numpy as np

class Metrics:
    """
    Class to compute various EEG metrics from PSD data.
    """

    def __init__(self, psd, freqs, channels):
        """
        Initialize the Metrics class.

        Parameters:
            psd (ndarray): PSD data (epochs × channels × frequencies).
            freqs (ndarray): Frequencies corresponding to PSD data.
            channels (list[str]): Channel names.
        """
        self.psd = psd
        self.freqs = freqs
        self.channels = channels

    def mean_power(self):
        """Compute mean PSD across epochs."""
        return np.mean(self.psd, axis=0)

    def variance_power(self):
        """Compute variance of PSD across epochs."""
        return np.var(self.psd, axis=0)

    def std_error_power(self):
        """Compute standard error of the mean PSD across epochs."""
        return np.std(self.psd, axis=0) / np.sqrt(self.psd.shape[0])

    def frequency_band_power(self, band):
        """
        Compute average power within a specified frequency band.

        Parameters:
            band (tuple): Frequency range as (low_freq, high_freq).

        Returns:
            ndarray: Mean power within the frequency band (epochs × channels).
        """
        low_freq, high_freq = band
        band_mask = (self.freqs >= low_freq) & (self.freqs <= high_freq)
        band_power = self.psd[:, :, band_mask].mean(axis=2)
        return band_power

    def aggregate_frequency_bands(self, bands):
        """
        Compute power across multiple standard EEG frequency bands.

        Parameters:
            bands (dict): Dictionary of bands with names and frequency ranges.
                          Example: {'alpha': (8, 12), 'beta': (13, 30)}

        Returns:
            dict: Mean band power (epochs × channels) for each band.
        """
        aggregated_powers = {}
        for band_name, freq_range in bands.items():
            aggregated_powers[band_name] = self.frequency_band_power(freq_range)
        return aggregated_powers

    def channel_power(self, channel_name):
        """
        Compute mean PSD across epochs for a specific channel.

        Parameters:
            channel_name (str): Name of the channel.

        Returns:
            ndarray: Mean PSD for the specified channel.
        """
        if channel_name not in self.channels:
            raise ValueError(f"Channel '{channel_name}' not found in channels.")
        channel_idx = self.channels.index(channel_name)
        return self.psd[:, channel_idx, :].mean(axis=0)

    def global_mean_power(self):
        """
        Compute the global mean PSD across epochs and channels.

        Returns:
            ndarray: Global mean PSD (frequencies,).
        """
        return self.psd.mean(axis=(0, 1))

    def summary_statistics(self):
        """
        Compute a set of descriptive summary statistics for PSD data.

        Returns:
            dict: Dictionary containing mean, variance, and std error.
        """
        return {
            'mean_power': self.mean_power(),
            'variance_power': self.variance_power(),
            'std_error_power': self.std_error_power()
        }

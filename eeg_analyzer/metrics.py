"""
Metrics
-------

Provides static or service-based methods for computing EEG metrics on PSD data.

Responsibilities:
- Compute alpha-band power and other frequency-band metrics.
- Normalize or transform PSDs (e.g., relative power, log power).
- Operate at the level of epoch × channel × frequency arrays.

Notes:
- Should be stateless and functional in style.
- Designed to be reusable across recordings, subjects, and datasets.
"""


import numpy as np

class Metrics:
    """
    Class to compute various EEG metrics from PSD data.
    """

    @staticmethod
    def mean_power(psd):
        """Compute mean PSD across epochs."""
        return np.mean(psd, axis=0)

    @staticmethod
    def variance_power(psd):
        """Compute variance of PSD across epochs."""
        return np.var(psd, axis=0)

    @staticmethod
    def std_error_power(psd):
        """Compute standard error of the mean PSD across epochs."""
        return np.std(psd, axis=0) / np.sqrt(psd.shape[0])

    @staticmethod
    def band_power(psd: np.ndarray, freqs: np.ndarray, band: tuple[float, float], operation: str = 'sum') -> np.ndarray:
        """
        Compute total or mean power within a frequency band using the specified aggregation.

        Parameters:
            psd (ndarray): PSD data (epochs × channels × frequencies), in μV²/Hz.
            freqs (ndarray): Frequencies corresponding to PSD data.
            band (tuple): Frequency range as (low_freq, high_freq).
            operation (str): Aggregation method:
                            'mean' → returns mean power spectral density (μV²/Hz),
                            'sum'  → returns total power (μV²), i.e. PSD × Δf summed over band.

        Returns:
            ndarray: Aggregated power (epochs × channels).
        """
        low, high = band
        band_idx = (freqs >= low) & (freqs <= high)

        if not np.any(band_idx):
            raise ValueError(f"No frequencies found in band range {band}")

        # Get frequency bin width assuming uniform spacing
        df = np.mean(np.diff(freqs))

        operation = operation.lower()
        if operation == 'mean':
            return psd[..., band_idx].mean(axis=-1)  # μV²/Hz
        elif operation == 'sum':
            return psd[..., band_idx].sum(axis=-1) * df  # μV²
        else:
            raise ValueError("Invalid operation. Use 'mean' or 'sum'.")

    @staticmethod
    def band_log(psd: np.ndarray, freqs: np.ndarray, band: tuple[float, float], operation: str = 'mean') -> np.ndarray:
        """
        Compute log-transformed values within a frequency band using the specified aggregation.
        The PSD data is first aggregated over the frequency band, then converted to log scale.

        Parameters:
            psd (ndarray): PSD data (epochs × channels × frequencies).
            freqs (ndarray): Frequencies corresponding to PSD data.
            band (tuple): Frequency range as (low_freq, high_freq).
            operation (str): Aggregation method ('mean' or 'sum').

        Returns:
            ndarray: Aggregated log-power (epochs × channels).
        """
        power_in_band = Metrics.band_power(psd, freqs, band, operation)
        return Metrics.log_transform(power_in_band)

    @staticmethod
    def aggregate_frequency_bands(psd, freqs, bands, operation='sum'):
        """
        Compute power across multiple standard EEG frequency bands.

        Parameters:
            psd (ndarray): PSD data (epochs × channels × frequencies).
            freqs (ndarray): Frequencies corresponding to PSD data.
            bands (dict): Dictionary of bands with names and frequency ranges.
                          Example: {'alpha': (8, 12), 'beta': (12, 30)}

        Returns:
            dict: Mean band power (epochs × channels) for each band.
        """
        return {
            band_name: Metrics.band_power(psd, freqs, freq_range, operation)
            for band_name, freq_range in bands.items()
        }

    @staticmethod
    def channel_power(psd, channels, channel_name):
        """
        Compute mean PSD across epochs for a specific channel.

        Parameters:
            psd (ndarray): PSD data (epochs × channels × frequencies).
            channels (list[str]): Channel names.
            channel_name (str): Name of the channel.

        Returns:
            ndarray: Mean PSD for the specified channel.
        """
        if channel_name not in channels:
            raise ValueError(f"Channel '{channel_name}' not found in channels.")
        channel_idx = channels.index(channel_name)
        return psd[:, channel_idx, :].mean(axis=0)

    @staticmethod
    def global_mean_power(psd):
        """
        Compute the global mean PSD across epochs and channels.

        Parameters:
            psd (ndarray): PSD data (epochs × channels × frequencies).

        Returns:
            ndarray: Global mean PSD (frequencies,).
        """
        return psd.mean(axis=(0, 1))

    @staticmethod
    def summary_statistics(psd):
        """
        Compute a set of descriptive summary statistics for PSD data.

        Parameters:
            psd (ndarray): PSD data (epochs × channels × frequencies).

        Returns:
            dict: Dictionary containing mean, variance, and std error.
        """
        return {
            'mean_power': Metrics.mean_power(psd),
            'variance_power': Metrics.variance_power(psd),
            'std_error_power': Metrics.std_error_power(psd)
        }
    
    @staticmethod
    def log_transform(psd: np.ndarray, epsilon: float = 1e-20) -> np.ndarray:
        """
        Convert PSD from µV²/Hz to natural logarithm scale using ln(psd).
        Clips values to prevent log of zero.
        """
        return np.log(np.maximum(psd, epsilon))  # Natural logarithm transformation to ln(microvolts squared per Hz)


    @staticmethod
    def alpha_power(psd: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """
        Compute the summed alpha power (8-12 Hz) for each epoch and channel.

        Parameters:
            psd (ndarray): PSD data (epochs × channels × frequencies).
            freqs (ndarray): Frequencies corresponding to PSD data.

        Returns:
            ndarray: Summed alpha power with shape (epochs × channels).
        """
        alpha_band = (8, 12)
        return Metrics.band_power(psd, freqs, alpha_band, operation='sum')
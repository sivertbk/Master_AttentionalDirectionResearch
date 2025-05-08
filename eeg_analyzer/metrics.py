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
        Compute power within a frequency band using the specified aggregation.

        Parameters:
            psd (ndarray): PSD data (epochs × channels × frequencies).
            freqs (ndarray): Frequencies corresponding to PSD data.
            band (tuple): Frequency range as (low_freq, high_freq).
            operation (str): Aggregation method ('mean' or 'sum').

        Returns:
            ndarray: Aggregated power (epochs × channels).
        """
        low, high = band
        band_idx = (freqs >= low) & (freqs <= high)

        if not np.any(band_idx):
            raise ValueError(f"No frequencies found in band range {band}")

        operation = operation.lower()
        if operation == 'mean':
            return psd[..., band_idx].mean(axis=-1)
        elif operation == 'sum':
            return psd[..., band_idx].sum(axis=-1)
        else:
            raise ValueError("Invalid operation. Use 'mean' or 'sum'.")

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
    def to_db(psd: np.ndarray, epsilon: float = 1e-20) -> np.ndarray:
        """
        Convert PSD from µV²/Hz to dB scale using 10 * log10(psd).
        Clips values to prevent log of zero.
        """
        return 10 * np.log10(np.maximum(psd, epsilon))

    @staticmethod
    def normalize_psd(psd: np.ndarray) -> np.ndarray:
        """
        Normalize PSD data using min-max scaling to range [0, 1].

        Normalization is applied across all epochs and channels for each frequency.

        Parameters:
            psd (ndarray): PSD data (epochs × channels × frequencies).

        Returns:
            ndarray: Min-max normalized PSD data (epochs × channels × frequencies).
        """
        psd_min = psd.min(axis=(0, 1), keepdims=True)
        psd_max = psd.max(axis=(0, 1), keepdims=True)
        return (psd - psd_min) / (psd_max - psd_min + 1e-20)

    @staticmethod
    def zscore_normalize_psd(psd: np.ndarray) -> np.ndarray:
        """
        Normalize PSD data using z-score normalization.

        Normalization is applied across all epochs and channels for each frequency.

        Parameters:
            psd (ndarray): PSD data (epochs × channels × frequencies).

        Returns:
            ndarray: Z-score normalized PSD data (epochs × channels × frequencies).
        """
        psd_mean = psd.mean(axis=(0, 1), keepdims=True)
        psd_std = psd.std(axis=(0, 1), keepdims=True)
        return (psd - psd_mean) / (psd_std + 1e-20)

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
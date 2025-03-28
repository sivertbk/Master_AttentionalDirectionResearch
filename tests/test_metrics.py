import unittest
import numpy as np
from eeg_analyzer.metrics import Metrics

class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Sample PSD data: 2 epochs, 3 channels, 5 frequencies
        self.psd = np.array([
            [[1, 2, 3, 0, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]],
            [[2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8]]
        ])
        self.freqs = np.array([1, 2, 3, 4, 5])
        self.channels = ['Cz', 'Pz', 'Fz']
        self.metrics = Metrics(self.psd, self.freqs, self.channels)

    def test_log_power(self):
        log_power = self.metrics.log_power()
        self.assertEqual(log_power.shape, self.psd.shape)
        self.assertTrue(np.all(log_power >= 0))  # Log of positive values
        self.assertTrue(np.all(np.isfinite(log_power)))  # No NaN or inf values

    def test_mean_power(self):
        mean_power = self.metrics.mean_power()
        self.assertEqual(mean_power.shape, (3, 5))  # Channels × Frequencies

    def test_variance_power(self):
        variance_power = self.metrics.variance_power()
        self.assertEqual(variance_power.shape, (3, 5))  # Channels × Frequencies

    def test_std_error_power(self):
        std_error_power = self.metrics.std_error_power()
        self.assertEqual(std_error_power.shape, (3, 5))  # Channels × Frequencies

    def test_frequency_band_power(self):
        band_power = self.metrics.frequency_band_power((2, 4))
        self.assertEqual(band_power.shape, (2, 3))  # Epochs × Channels

    def test_aggregate_frequency_bands(self):
        bands = {'alpha': (2, 4), 'beta': (4, 5)}
        aggregated = self.metrics.aggregate_frequency_bands(bands)
        self.assertEqual(set(aggregated.keys()), {'alpha', 'beta'})
        for band in aggregated.values():
            self.assertEqual(band.shape, (2, 3))  # Epochs × Channels

    def test_channel_power(self):
        channel_power = self.metrics.channel_power('Cz')
        self.assertEqual(channel_power.shape, (5,))  # Frequencies

    def test_global_mean_power(self):
        global_mean = self.metrics.global_mean_power()
        self.assertEqual(global_mean.shape, (5,))  # Frequencies

    def test_summary_statistics(self):
        summary = self.metrics.summary_statistics()
        self.assertIn('mean_power', summary)
        self.assertIn('variance_power', summary)
        self.assertIn('std_error_power', summary)
        self.assertEqual(summary['mean_power'].shape, (3, 5))  # Channels × Frequencies

    def test_indexing(self):
        epoch = self.metrics[0]
        self.assertTrue(np.array_equal(epoch, self.psd[0]))

    def test_iteration(self):
        epochs = list(iter(self.metrics))
        self.assertEqual(len(epochs), len(self.psd))
        for i, epoch in enumerate(epochs):
            self.assertTrue(np.array_equal(epoch, self.psd[i]))

    def test_invalid_channel(self):
        with self.assertRaises(ValueError):
            self.metrics.channel_power('InvalidChannel')

    def test_out_of_range_index(self):
        with self.assertRaises(IndexError):
            _ = self.metrics[len(self.psd)]

if __name__ == '__main__':
    unittest.main()

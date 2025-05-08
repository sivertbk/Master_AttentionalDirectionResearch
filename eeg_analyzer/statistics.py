"""
Statistics
----------

Implements statistical tests and models for analyzing EEG data.

Responsibilities:
- Perform non-parametric testing (e.g., Mann-Whitney U, permutation cluster tests).
- Fit linear models and mixed-effects models across conditions or groups.
- Return standardized results for downstream visualization and reporting.

Notes:
- Inputs are typically epoch-level metric arrays with associated metadata (e.g., labels, subject IDs).
- Designed for composability: results should be easy to pass into visualizers.
"""

from scipy.stats import mannwhitneyu


class Statistics:
    
    def __init__(self, data):
        self.data = data

    def mean(self):
        return sum(self.data) / len(self.data)

    def median(self):
        sorted_data = sorted(self.data)
        n = len(sorted_data)
        if n % 2 == 0:
            return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        return sorted_data[n // 2]

    def mode(self):
        counts = {}
        for value in self.data:
            if value in counts:
                counts[value] += 1
            else:
                counts[value] = 1
        max_count = max(counts.values())
        return [key for key, value in counts.items() if value == max_count]

    def variance(self):
        mean = self.mean()
        return sum((x - mean) ** 2 for x in self.data) / len(self.data)

    def standard_deviation(self):
        return self.variance() ** 0.5

    def mann_whitney_u_test(self, group1, group2, alternative="greater"):
        """
        Perform a one-tailed Mann-Whitney U test per channel.

        Parameters:
        - group1: List of values for group 1 (e.g., OT recordings).
        - group2: List of values for group 2 (e.g., MW recordings).
        - alternative: Hypothesis ('greater', 'less', or 'two-sided').

        Returns:
        - p_values: Dictionary with channel names as keys and p-values as values.
        """
        p_values = {}
        for channel in group1.keys():
            u_stat, p_value = mannwhitneyu(group1[channel], group2[channel], alternative=alternative)
            p_values[channel] = p_value
        return p_values
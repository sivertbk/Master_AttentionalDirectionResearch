

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
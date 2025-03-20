import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, data):
        self.data = data

    def plot(self):
        plt.plot(self.data)
        plt.show()
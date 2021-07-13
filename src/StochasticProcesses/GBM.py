import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class GBM:
    def __init__(self, initial_value, drift, volatility, T, dt, simulation_count):
        self.S0 = initial_value
        self.mu = drift
        self.sigma = volatility
        self.dt = dt
        self.T = T
        self.time_step_count = np.int(self.T / self.dt) + 1
        self.simulation_count = simulation_count
        self.paths = []

    # time steps can be tricky if T/dt is not a round number
    # TODO: use linspace instead
    def generate_paths(self):
        self.paths = np.zeros([self.simulation_count, self.time_step_count])
        self.paths[:, 0] = self.S0
        for i in range(1, self.time_step_count):
            z = np.random.standard_normal(self.simulation_count)
            self.paths[:, i] = self.paths[:, i - 1] \
                * np.exp((self.mu - 0.5 * self.sigma ** 2)
                            * self.dt + self.sigma * np.sqrt(self.dt) * z)

    def plot_paths(self):
        sorted_indices_of_averages = np.argsort(np.average(self.paths, 1))
        sorted_paths = self.paths[sorted_indices_of_averages].T
        t = np.linspace(0, self.T, self.time_step_count)
        sns.set_palette(sns.cubehelix_palette(self.simulation_count, start=.5, rot=-.75))
        plt.plot(t, sorted_paths)
        plt.show()

    def get_time_steps(self):
        return np.linspace(0, self.T, self.time_step_count)

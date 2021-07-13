import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class GBM:
    def __init__(self, initial_value, drift, volatility, time_to_maturity, time_step_size, simulation_count):
        """
        Returns geometric Brownian motion (GBM) class.

        :param initial_value: Initial asset value.
        :param drift: Drift.
        :param volatility: Volatility.
        :param time_to_maturity: Time to maturity.
        :param time_step_size: Time step size i.e. number of time steps = T/dt + 1
        :param simulation_count: Number of simulations to perform.
        """
        self.initial_value = initial_value
        self.drift = drift
        self.sigma = volatility
        self.dt = time_step_size
        self.time_to_maturity = time_to_maturity
        self.time_step_count = int(self.time_to_maturity / self.dt) + 1
        self.simulation_count = simulation_count
        self.paths = []

    # time steps can be tricky if T/dt is not a round number
    # TODO: use linspace instead
    def generate_paths(self):
        self.paths = np.zeros([self.simulation_count, self.time_step_count])
        self.paths[:, 0] = self.initial_value
        for i in range(1, self.time_step_count):
            z = np.random.standard_normal(self.simulation_count)
            self.paths[:, i] = self.paths[:, i - 1] \
                               * np.exp((self.drift - 0.5 * self.sigma ** 2)
                                        * self.dt + self.sigma * np.sqrt(self.dt) * z)

    def plot_paths(self):
        indices_sorted_by_path_averages = np.argsort(np.average(self.paths, 1))
        sorted_paths = np.transpose(self.paths[indices_sorted_by_path_averages])
        t = np.linspace(0, self.time_to_maturity, self.time_step_count)
        sns.set_palette(sns.cubehelix_palette(self.simulation_count, start=.5, rot=-.75))
        fig, ax1 = plt.subplots()

        # plot GBM asset price paths
        ax1.set_xlabel('Time (Years)')
        ax1.set_ylabel('Asset Price')
        ax1.set_xlim([0, self.time_to_maturity])
        ax1.plot(t, sorted_paths)

        plt.show()

    def get_time_steps(self):
        return np.linspace(0, self.time_to_maturity, self.time_step_count)

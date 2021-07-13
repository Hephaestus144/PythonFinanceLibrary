import numpy as np

from src.Finance.OptionPricing.OptionStyle import OptionStyle
from src.Finance.StochasticProcesses.GBM import GBM


class AsianOption:
    # assuming averaging period lasts till maturity
    def __init__(self, initial_value, strike, drift, volatility, T, averaging_period_start, option_style):
        self.S0 = initial_value
        self.K = strike
        self.mu = drift
        self.sigma = volatility
        self.T = T
        self.T_avg_start = averaging_period_start
        self.option_style = option_style

    def monte_carlo_price(self, simulation_count, time_step_size):
        gbm = GBM(self.S0, self.mu, self.sigma, self.T, time_step_size, simulation_count)
        gbm.generate_paths()
        time_steps = gbm.get_time_steps()
        T_avg_start_index = np.where(time_steps >= self.T_avg_start)[0][0]
        if T_avg_start_index == 0:
            T_avg_start_index = 1
        prices = np.mean(gbm.paths[:, T_avg_start_index:], 1)
        if self.option_style is OptionStyle.CALL:
            prices = np.exp(-self.mu*self.T)*np.maximum(prices - self.K, 0)
        else:
            prices = np.exp(-self.mu * self.T) * np.maximum(self.K - prices, 0)
        price = np.mean(prices)
        stddev = np.std(prices) / np.sqrt(simulation_count)
        return [price, stddev]

    # Turnbull-Wakeman methods
    # h = time_step_size (between averaging periods)
    # n = number of sampling points
    def turnbull_wakeman_price(self, n):
        h = self.T / n
        meanG = np.log(self.S0) + (self.mu - 0.5*self.sigma**2)*(self.T + h)/2
        varG = self.sigma**2 * h * (2 * n + 1) * (n + 1) / (6 * n)
        meanS = np.zeros(n)
        varS = np.zeros(n)
        covSG = np.zeros(n)
        for i in range(1, n):
            meanS[i - 1] = np.log(self.S0) + (self.mu - 0.5 * self.sigma**2) * i * h
            varS[i - 1] = self.sigma**2 * i * h
            covSG[i - 1] = self.sigma**2 * h * ((2 * n + 1) * i - i**2) / (2 * n)

        result = 0
        for i in range(1, n):
            result += np.exp(meanS[i - 1] + (self.mu - meanG)*covSG[i - 1] / varG
                             + (varS[i - 1] - covSG[i - 1]**2 / varG) / 2)
        return result

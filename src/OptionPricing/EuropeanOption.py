import numpy as np
from scipy.stats import norm

from src.Finance.OptionPricing.OptionStyle import OptionStyle
from src.Finance.StochasticProcesses.GBM import GBM


class EuropeanOption:
    def __init__(self, initial_value, strike, drift, volatility, T, option_style):
        self.S0 = initial_value
        self.K = strike
        self.mu = drift
        self.sigma = volatility
        self.T = T
        self.option_style = option_style

    def black_scholes_price(self):
        d1 = (np.log(self.S0 / self.K) + (self.mu + 0.5 * self.sigma ** 2) * self.T) \
             / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        if self.option_style is OptionStyle.CALL:
            return self.S0 * norm.cdf(d1) - self.K * np.exp(-self.mu * self.T) * norm.cdf(d2)
        else:
            return -self.S0 * norm.cdf(-d1) + self.K * np.exp(-self.mu * self.T) * norm.cdf(-d2)

    # time_step_size should be irrelevant for a European option
    def monte_carlo_price(self, simulation_count, time_step_size, return_tenors_and_prices_paths=False):
        gbm = GBM(self.S0, self.mu, self.sigma, self.T, time_step_size, simulation_count)
        gbm.generate_paths()
        if self.option_style == OptionStyle.CALL:
            prices = np.exp(-self.mu * self.T) \
               * np.maximum(gbm.paths[:, -1] - self.K, 0)
        else:
            prices = np.exp(-self.mu * self.T) \
                     * np.maximum(self.K - gbm.paths[:, -1], 0)
        if return_tenors_and_prices_paths:
            time_steps = gbm.get_time_steps()
            prices = np.exp(-self.mu * time_steps) \
                * np.maximum(gbm.paths - self.K, 0)
            return [gbm.get_time_steps(), prices]
        else:
            price = np.mean(prices)
            stddev = np.std(prices)/np.sqrt(simulation_count)
            return [price, stddev]


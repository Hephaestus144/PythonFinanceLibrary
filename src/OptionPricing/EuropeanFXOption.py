import numpy as np
from scipy.stats import norm

from src.Finance.OptionPricing.OptionStyle import OptionStyle
from src.Finance.StochasticProcesses.GBM import GBM
from src.Finance.Curves.InterestRateCurve import InterestRateCurve

class EuropeanFXOption:
    def __init__(self, initial_value, strike, domestic_curve: InterestRateCurve,
                 foreign_curve: InterestRateCurve, volatility, T, option_style):
        self.S0 = initial_value
        self.K = strike
        self.domestic_curve = domestic_curve
        self.foreign_curve = foreign_curve
        self.sigma = volatility
        self.T = T
        self.option_style = option_style

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


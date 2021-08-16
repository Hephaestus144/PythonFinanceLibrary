import numpy as np
from src.Curves.Curve import Curve
from src.Swaps.Frequency import Frequency


class InterestRateSwap:
    """
    A class for vanilla interest rate swaps with the following assumptions:
    * Pay and receive frequencies match perfectly i.e. there is no basis
    * Pay and receive frequencies are 'perfect' fractions of years e.g. 0.25, 0.5, 1.0.
    """
    def __init__(self,
                 notional: float,
                 strike: float,
                 start_tenor: float,
                 maturity_tenor: float,
                 payment_frequency: Frequency):
        self.notional: float = notional
        self.strike: float = strike
        self.start_tenor: float = start_tenor
        self.maturity_tenor: float = maturity_tenor
        self.payment_frequency: Frequency = payment_frequency

        # The time step between payment dates e.g. 0.25 = 3m.
        self.time_step = 0
        if self.payment_frequency == Frequency.QUARTERLY:
            self.time_step = 0.25
        elif self.payment_frequency == Frequency.SEMIANNUALLY:
            self.time_step = 0.50
        elif self.payment_frequency == Frequency.ANNUALLY:
            self.time_step = 1.00

        self.payment_tenors: np.ndarray = np.arange(self.start_tenor + self.time_step,
                                                    self.maturity_tenor,
                                                    self.time_step)

    def compute_fair_swap_rate(self, curve: Curve):
        dfs: np.ndarray = curve.get_discount_factors(self.payment_tenors)
        forward_rates: np.ndarray\
            = curve.get_forward_rates(
                np.append(self.start_tenor, self.payment_tenors[:-1]),
                self.payment_tenors)

        numerator: float = 0
        denominator: float = 0

        for i, df in enumerate(dfs):
            numerator += forward_rates[i] * self.time_step * df
            denominator += self.time_step * df

        return numerator / denominator




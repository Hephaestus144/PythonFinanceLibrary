import numpy as np
from src.Curves.Curve import Curve
from src.Swaps.Frequency import Frequency


class InterestRateSwap:
    """
    A class for vanilla interest rate swap i.e. it cannot accommodate basis swaps.
    """
    def __init__(self,
                 notional: float,
                 strike: float,
                 tenor: float,
                 payment_frequency: Frequency,
                 curve: Curve):
        self.notional: float = notional
        self.strike: float = strike
        self.maturity_tenor: float = tenor
        self.payment_frequency: Frequency = payment_frequency
        self.curve: Curve = curve

        # The time step between payment dates e.g. 0.25 = 3m.
        time_step = 0
        if self.payment_frequency == Frequency.QUARTERLY:
            time_step = 0.25

        self.payment_tenors: np.ndarray = np.arange(time_step, self.maturity_tenor, time_step)




import numpy as np


class HullWhite:
    """
    Notes:
        • According to https://quant.stackexchange.com/questions/10194/simulating-the-short-rate-in-the-hull-white-model
          you don't need option prices except for generalised Hull-White
        • Apparently you can calibrate theta as piece-wise constant.
    """
    def __init__(self, alpha, sigma, initial_curve):
        self.alpha = alpha
        self.sigma = sigma
        self.initial_curve = initial_curve

    def b_function(self, start_time: float, end_time: float) -> float:
        """
        Calculates the value of the classic 'B' function commonly associated with Hull-White.

        :param start_time: The start time.
        :type start_time: float
        :param end_time: The end time.
        :type end_time: float
        :returns: Value of B function for Hull-White (see Green, Shreve, et al.)
        :rtype: float
        """
        return (1/self.alpha) * (1 - np.exp(-1 * self.alpha * (end_time - start_time)))

    def swaption_pricing_sigma(self, swaption_expiry: float, swap_cashflow_tenors: np.ndarray):
        """
        This implements the swaption pricing formula for the volatility as per formula 16.95 of Green.
        """

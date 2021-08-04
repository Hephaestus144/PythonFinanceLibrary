import numpy as np
from scipy.interpolate import interpolate
from scipy.stats import norm


class HullWhite:
    """
    Notes:
        • According to https://quant.stackexchange.com/questions/10194/simulating-the-short-rate-in-the-hull-white-model
          you don't need option prices except for generalised Hull-White
        • Apparently you can calibrate theta as piece-wise constant.
    """
    def __init__(self, alpha, sigma_tenors, sigmas, initial_curve):
        self.alpha = alpha
        self.sigma_tenors = sigma_tenors
        self.sigmas = sigmas
        if len(sigmas) != 1:
            self.sigma_interpolator =\
                interpolate.interp1d(
                    self.sigma_tenors,
                    self.sigmas,
                    kind='previous',
                    fill_value=(self.sigmas[0], self.sigmas[-1]),
                    bounds_error=False)
        else:
            self.sigma_interpolator = None

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

    def swaption_pricing_sigma(self, strike: float, swaption_expiry: float, swap_cashflow_tenors: np.ndarray):
        """
        This implements the swaption pricing formula for the volatility as per formula 16.95 of Green.
        """
        b = np.zeros(len(swap_cashflow_tenors))
        numerator = 0
        denominator = 0
        b[0] = 1
        b[-1] = 1 + strike * (swap_cashflow_tenors[-1] - swap_cashflow_tenors[-2])
        for i in range(1, len(swap_cashflow_tenors) - 1):
            b[i] = strike * (swap_cashflow_tenors[i] - swap_cashflow_tenors[i - 1])

        for i in range(0, len(swap_cashflow_tenors)):
            df = self.initial_curve.get_discount_factors(swap_cashflow_tenors[i])
            numerator +=\
                b[i] * (self.b_function(swaption_expiry, swap_cashflow_tenors[i]) -
                        self.b_function(swaption_expiry, swap_cashflow_tenors[-1])) * \
                df
            denominator += b[i] * df

        return self.interpolate_sigma(swaption_expiry) * numerator / denominator

    def interpolate_sigma(self, time: float):
        if self.sigma_interpolator is None:
            return self.sigmas[0]
        else:
            return self.sigma_interpolator(time)

    def swaption_pricer(self, strike, swaption_expiry: float, h, swap_cashflow_tenors):
        v = self.swaption_pricing_sigma(strike, swaption_expiry, swap_cashflow_tenors)
        d1 = np.log(h) / v + 0.5 * v
        d2 = d1 - v
        df = self.initial_curve.get_discount_factors(swap_cashflow_tenors[-1])
        return df * (h * norm.cdf(d1) - norm.cdf(d2))

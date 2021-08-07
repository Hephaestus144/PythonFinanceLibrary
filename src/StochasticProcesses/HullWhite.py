import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interpolate
from scipy.stats import norm
from src.Curves.Curve import Curve


class HullWhite:
    """
    A class for the Hull-White stochastic process.
    :math:`dr(t) = (\\theta(t) - \\alpha r(t))dt + \\sigma(t) dW(t)`
    """

    def __init__(self, alpha: float, sigma_tenors: np.ndarray, sigmas: np.ndarray, initial_curve: Curve):
        """
        Constructor for the Hull-White process.

        :param alpha: The standard mean reversion speed, commonly denoted :math:`\\alpha`.
        :type alpha: float
        :param sigma_tenors: The corresponding tenors for the sigma_tenors parameters.
        :type sigma_tenors: np.ndarray
        :param sigmas: The standard volatility of the Hull-White process, commonly denoted :math:`\\sigma(t)`.
        :type sigmas: np.ndarray
        """
        # TODO: Add theta
        self.alpha = alpha
        self.sigma_tenors = sigma_tenors
        self.sigmas = sigmas
        if len(sigmas) != 1:
            self.sigma_interpolator = \
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
        return (1 / self.alpha) * (1 - np.exp(-1 * self.alpha * (end_time - start_time)))

    def interpolate_sigma(self, time: float):
        """
        Interpolates the Hull-White :math:`\\sigma` parameter (piecewise constant from the left).
        If the :math:`\\sigma` parameter is single valued it returns the single value.

        :param time: The time point at which to interpolate.
        :type time: float
        :returns: Interpolated :math:`\\sigma` value.
        :rtype: float
        """
        if self.sigma_interpolator is None:
            return self.sigmas[0]
        else:
            return self.sigma_interpolator(time)

    def swaption_pricing_vol(
            self,
            time: float,
            strike: float,
            swaption_expiry: float,
            swap_cashflow_tenors: np.ndarray) -> float:
        """
        • This implements the swaption pricing formula for the volatility as per formula 16.95 of Green. Denoted
        :math:`\\Sigma(t)^2`.
        • This is not to be confused with the :math:`\\sigma` Hull-White parameter.

        :param time: The tenor for which we're calculating the volatility.
        :type time: float
        :param strike: Swaption strike.
        :type strike: float
        :param swaption_expiry: Expiry tenor of the swaption.
        :type swaption_expiry: float
        :param swap_cashflow_tenors: Tenors for the swap underlying the swaption. These need to be greater than
        swaption_expiry.
        :type swap_cashflow_tenors: np.ndarray
        :returns: Value of swaption pricing vol in terms of Hull-White parameters :math:`\\alpha` &:math:`\\sigma`.
        :rtype: float
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
            numerator += \
                b[i] * (self.b_function(time, swap_cashflow_tenors[i]) -
                        self.b_function(time, swaption_expiry)) * \
                df
            denominator += b[i] * df

        return (self.interpolate_sigma(time) * numerator / denominator) ** 2

    def weighted_strike(self, strike: float, swaption_expiry: float, swap_cashflow_tenors: np.ndarray) -> float:
        """
        Calculates the time weighted strike denoted :math:`H(t)` in Green.

        :param strike: The original swaption strike.
        :type strike: float
        :param swaption_expiry: The expiry of the swaption.
        :type swaption_expiry: float
        :param swap_cashflow_tenors: The tenors of the swap underlying the swaption.
        :type swap_cashflow_tenors: np.ndarray
        :returns: The time weighted strike.
        :rtype: float
        """
        b = np.zeros(len(swap_cashflow_tenors))
        b[0] = 1
        b[-1] = 1 + strike * (swap_cashflow_tenors[-1] - swap_cashflow_tenors[-2])
        for i in range(1, len(swap_cashflow_tenors) - 1):
            b[i] = strike * (swap_cashflow_tenors[i] - swap_cashflow_tenors[i - 1])

        result = 0
        for i in range(1, len(swap_cashflow_tenors)):
            df = self.initial_curve.get_discount_factors(swap_cashflow_tenors[i])
            result += b[i]*(swap_cashflow_tenors[i] - swap_cashflow_tenors[i-1]) * df

        result /= self.initial_curve.get_discount_factors(swaption_expiry)
        return result

    def swaption_pricer(self, strike: float, swap_cashflow_tenors: np.ndarray) -> float:
        """
        This implements the swaption pricer using Hull-White parameters :math:`\\alpha` & :math:`\\sigma` as per
        formula (16.96) of Green.

        :param strike: Swaption strike.
        :type strike: float
        :param swap_cashflow_tenors: The tenors of the swap cashflows.
        :type swap_cashflow_tenors: np.ndarray
        :returns: The price of a swaption.
        :rtype: float
        """
        h0 = self.weighted_strike(strike, swap_cashflow_tenors)
        v = np.sqrt(quad(self.swaption_pricing_vol, 0, swap_cashflow_tenors[-1], args=(strike, swap_cashflow_tenors)))
        print(f'\nv: {v}\n')
        d1 = np.log(h0) / v[0] + 0.5 * v[0]
        d2 = d1 - v[0]
        df = self.initial_curve.get_discount_factors(swap_cashflow_tenors[-1])
        return df * (h0 * norm.cdf(d1) - norm.cdf(d2))


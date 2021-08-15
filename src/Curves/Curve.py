import math
import numpy as np
from scipy.interpolate import interpolate


class Curve:
    """
    Base class from which other curves inherit.
    """
    def __init__(self, **kwargs):
        """
        Constructor for base curve.
        Set zero_rates to single value to create flat curve.
        """
        self.tenors: np.ndarray = kwargs.pop('tenors', None)
        self.discount_factors: np.ndarray = kwargs.pop('discount_factors', None)
        self.zero_rates: np.ndarray = kwargs.pop('zero_rates', None)
        self.interpolation_method: str = kwargs.pop('interpolation_method', None)

        if self.zero_rates is None:
            self.zero_rates = -1 * np.log(self.discount_factors) / self.tenors

        if self.discount_factors is None:
            self.discount_factors = np.exp(-1 * np.array(self.zero_rates) * np.array(self.tenors))

        if not self.tenors.__contains__(0):
            self.tenors = np.append(0, self.tenors)
            self.discount_factors = np.append(1, self.discount_factors)

        if self.interpolation_method is None:
            self.interpolation_method = 'linear'

        self.discount_factor_interpolator =\
            interpolate.interp1d(
                self.tenors,
                self.discount_factors,
                kind=self.interpolation_method,
                fill_value=(self.discount_factors[0], self.discount_factors[-1]),
                bounds_error=False)

    def get_discount_factors(self, tenors: np.ndarray) -> np.ndarray:
        """
        Returns discount factors for a list of tenor(s).

        Perform linear interpolation and flat extrapolation.
        :param tenors: Tenor(s) for which to interpolate.
        :type tenors: np.ndarray
        :return: Array of discount factors.
        :rtype: np.ndarray
        """
        # A single zero rate implies the curve is flat.
        if len(self.zero_rates) == 1:
            return np.exp(-1 * self.zero_rates[0] * tenors)
        else:
            return self.discount_factor_interpolator(tenors)

    def get_forward_discount_factors(self, start_tenor: float, end_tenors: np.ndarray) -> np.ndarray:
        """
        Calculates forward discount factors.

        :param start_tenor: The date in the future towards which we are discounting.
        :type start_tenor: float
        :param end_tenors: The dates, after the start date, from which we are discounting back to the start date.
        :type end_tenors: np.ndarray
        :returns: Numpy array of discount factors.
        :rtype: np.ndarray
        """
        initial_df: float = self.get_discount_factors(start_tenor)
        end_dfs = self.get_discount_factors(end_tenors)
        return end_dfs/initial_df

    def get_forward_rates(self, start_points: np.ndarray, end_points: np.ndarray) -> np.ndarray:
        """
        Calculates forward rates (including zero rates which are just a special case).

        :param start_points: The starting time points for the forward rates.
        :type start_points: np.ndarray
        :param end_points: The end time points for the forward rates.
        :type end_points: np.ndarray
        :return: Array of forward rates.
        :rtype: np.ndarray
        """
        forward_rates: np.ndarray = np.zeros(len(start_points))
        start_discount_factors: np.ndarray = self.get_discount_factors(start_points)
        end_discount_factors: np.ndarray = self.get_discount_factors(end_points)
        for i in range(0, len(start_points)):
            forward_rates[i] = \
                1 / (end_points[i] - start_points[i]) * math.log(start_discount_factors[i] / end_discount_factors[i])
        return forward_rates

    def get_zero_rates(self, tenors: np.ndarray) -> np.ndarray:
        dfs: np.ndarray = self.get_discount_factors(tenors)
        zero_rates: np.ndarray = np.zeros(len(dfs))
        for i, df in enumerate(dfs):
            if tenors[i] == 0:
                zero_rates[i] = self.zero_rates[0]
            else:
                zero_rates[i] = -1 * np.log(dfs[i]) / tenors[i]
        return zero_rates

    def get_first_order_derivative_of_zero_rates(self, tenors: np.ndarray, delta_t: float = 0.0001) -> np.ndarray:
        tenors_plus_delta_t: np.ndarray = tenors + delta_t
        start_points = np.zeros(len(tenors))
        forward_rates: np.ndarray = self.get_forward_rates(start_points, tenors)
        forward_rates_plus_delta: np.ndarray = self.get_forward_rates(start_points, tenors_plus_delta_t)
        return (forward_rates_plus_delta - forward_rates) / delta_t

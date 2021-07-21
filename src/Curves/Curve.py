import math

import numpy as np


class Curve:
    """
    Base class from which other curves inherit.
    """
    def __init__(self, **kwargs):
        self.tenors: np.ndarray = kwargs.pop('tenors', None)
        self.discount_factors: np.ndarray = kwargs.pop('discount_factors', None)

        if not self.tenors.__contains__(0):
            self.tenors = np.append(0, self.tenors)
            self.discount_factors = np.append(1, self.discount_factors)

    def get_discount_factors(self, tenors: np.ndarray) -> np.ndarray:
        """
        Returns discount factors for a list of tenor(s).

        Perform linear interpolation and flat extrapolation.
        :param tenors: Tenor(s) for which to interpolate.
        :type tenors: np.ndarray
        :return: Array of discount factors.
        :rtype: np.ndarray
        """
        # TODO: Make interpolation configurable.
        # TODO: Make extrapolation configurable.
        return np.interp(tenors, self.tenors, self.discount_factors)

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

        forward_rates: np.ndarray = np.array([])
        start_discount_factors: np.ndarray = self.get_discount_factors(start_points)
        end_discount_factors: np.ndarray = self.get_discount_factors(end_points)
        for i in range(0, len(start_points)):
            forward_rate = 1 / (end_points[i] - start_points[i]) *\
                           math.log(start_discount_factors[i] / end_discount_factors[i])
            forward_rates = np.append(forward_rates, forward_rate)

        return forward_rates

    def get_first_order_derivative_of_zero_rates(self, tenors: np.ndarray) -> np.ndarray:
        delta_t: float = 0.0001
        tenors_plus_delta_t: np.ndarray = tenors + delta_t
        start_points = np.zeros(len(tenors))
        forward_rates: np.ndarray = self.get_forward_rates(start_points, tenors)
        forward_rates_plus_delta: np.ndarray = self.get_forward_rates(start_points, tenors_plus_delta_t)
        return (forward_rates_plus_delta - forward_rates) / delta_t

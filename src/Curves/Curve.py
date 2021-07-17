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

import numpy as np


class Curve:
    """
    Base class from which other curves inherit.
    """
    def __init__(self, **kwargs: list(float)):
        self.tenors: list(float) = kwargs.pop('tenors', None)
        self.discount_factors: list(float) = kwargs.pop('discount_factors', None)

        if not self.tenors.__contains__(0):
            self.tenors = np.append(0, self.tenors)
            self.discount_factors = np.append(1, self.discount_factors)

    def get_discount_factors(self, tenors: float | list(float)) -> float | list(float):
        """
        Returns discount factors for a list of tenor(s).
        :param tenors: Tenor(s) for which to interpolate.
        :type tenors: float | list(float)
        :return:
        :rtype: float | list(float)
        """
        # TODO: Make interpolation configurable.
        return np.interp(tenors, self.tenors, self.discount_factors)

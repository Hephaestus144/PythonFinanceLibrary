import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.Finance.Curves.RateConvention import RateConvention


class InterestRateCurve:
    def __init__(self, **kwargs):
        self.tenors = kwargs.pop('tenors', None)
        self.discount_factors = kwargs.pop('discount_factors', None)

        if self.discount_factors is None:
            self.rates = kwargs.pop('rates')
            self.rate_convention = kwargs.pop('rate_convention')
            if self.rate_convention == RateConvention.NACC:
                self.discount_factors = np.exp(-self.rates * self.tenors)
            elif self.rate_convention == RateConvention.NACM:
                self.discount_factors = (1 + self.rates / 12) ** (-12 * self.tenors)
            elif self.rate_convention == RateConvention.NACQ:
                self.discount_factors = (1 + self.rates / 4) ** (-4 * self.tenors)
            elif self.rate_convention == RateConvention.NACS:
                self.discount_factors = (1 + self.rates / 2) ** (-2 * self.tenors)
            elif self.rate_convention == RateConvention.NACA:
                self.discount_factors = (1 + self.rates) ** (-1 * self.tenors)

        if not self.tenors.__contains__(0):
            self.tenors = np.append(0, self.tenors)
            self.discount_factors = np.append(1, self.discount_factors)

    def get_discount_factors(self, tenors):
        return np.interp(tenors, self.tenors, self.discount_factors)

    def get_zero_rates(self, tenors, rate_convention):
        discount_factors = self.get_discount_factors(tenors)
        if rate_convention == RateConvention.NACC:
            return -np.log(discount_factors) / tenors
        elif rate_convention == RateConvention.NACM:
            # df = (1 + r/12)^12t
            # r = 12*[df^(-1/12t) - 1]
            return 12 * (discount_factors ** (-1 / (12 * tenors)) - 1)
        elif rate_convention == RateConvention.NACQ:
            # df = (1 + r/4)^4t
            # r = 4*[df^(-1/4t) - 1]
            return 4 * (discount_factors ** (-1 / (4 * tenors)) - 1)
        elif rate_convention == RateConvention.NACS:
            # df = (1 + r/2)^2t
            # r = 2*[df^(-1/2t) - 1]
            return 2 * (discount_factors ** (-1 / (2 * tenors)) - 1)
        elif rate_convention == RateConvention.NACA:
            # df = (1 + r)^t
            # r = 2*[df^(-1/t) - 1]
            return discount_factors ** (-1 / tenors) - 1
        else:
            return "Invalid rate convention"

    # assumes NACC rates for now
    def get_forward_rates(self, start_tenors, end_tenors):
        start_discount_factors = self.get_discount_factors(start_tenors)
        end_discount_factors = self.get_discount_factors(end_tenors)
        t = end_tenors - start_tenors
        return (1/t)*np.log(start_discount_factors/end_discount_factors)

    def plot_curve(self, values_to_plot='discount_factors'):
        if values_to_plot.lower() == 'discount_factors':
            plt.plot(self.tenors, self.discount_factors)
        elif values_to_plot.lower() == 'rates':
            plt.plot(self.tenors, self.rates)
        plt.show()


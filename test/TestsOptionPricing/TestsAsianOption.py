import unittest

from src.Finance.OptionPricing.AsianOption import *
from src.Finance.OptionPricing.OptionStyle import *


# Asian option pricing examples taken from :
# Numerical Methods in Finance & Economics by Paolo Brandimarte pg 456 & 461


# TODO: asian should be cheaper than European;
# TODO: asian volatility should be lower than European
# Asian option price is sensitive to the number of sample points.
# As you increase the number of sample points the price seems to decrease
# as you're including more points at inception... this seems dubious
# Geometric Asian options are cheaper than Arithmetic ones because
# the geometric mean is less than the arithmetic mean.
class TestsAsianOption(unittest.TestCase):
    def test_monte_carlo_call_pricing_with_short_tenor(self):
        option = AsianOption(50, 50, 0.1, 0.4, 5/12, 0, OptionStyle.CALL)
        [price, stddev] = option.monte_carlo_price(10000, 5/(12*5))
        self.assertTrue(3.9605 - 3*stddev < price < 3.9605 + 3*stddev)

    def test_monte_carlo_call_pricing_with_long_tenor(self):
        option = AsianOption(50, 50, 0.1, 0.4, 2, 0, OptionStyle.CALL)
        [price, stddev] = option.monte_carlo_price(10000, 5/(12*24))
        self.assertTrue(8.3424 - 3*stddev < price < 8.3424 + 3*stddev)

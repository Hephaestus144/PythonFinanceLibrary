import unittest

from src.Finance.OptionPricing.EuropeanOption import EuropeanOption
from src.Finance.OptionPricing.OptionStyle import OptionStyle


# European option price example taken from :
# Numerical Methods in Finance & Economics by Paolo Brandimarte pg 113


class TestsEuropeanOption(unittest.TestCase):
    def test_basic_call_black_scholes(self):
        option = EuropeanOption(50, 52, 0.1, 0.4, 5 / 12, OptionStyle.CALL)
        self.assertAlmostEqual(option.black_scholes_price(), 5.1911, 4)

    def test_basic_put_black_scholes(self):
        option = EuropeanOption(50, 52, 0.1, 0.4, 5 / 12, OptionStyle.PUT)
        self.assertAlmostEqual(option.black_scholes_price(), 5.0689, 4)

    def test_monte_carlo_call_pricing(self):
        option = EuropeanOption(50, 52, 0.1, 0.4, 5 / 12, OptionStyle.CALL)
        [price, stddev] = option.monte_carlo_price(1000, 5/12)
        self.assertTrue(5.1911 - 2*stddev < price < 5.1911 + 2*stddev)

    def test_monte_carlo_put_pricing(self):
        option = EuropeanOption(50, 52, 0.1, 0.4, 5 / 12, OptionStyle.PUT)
        [price, stddev] = option.monte_carlo_price(1000, 5/12)
        print(price)
        self.assertTrue(5.0689 - 3*stddev < price < 5.0689 + 3*stddev)

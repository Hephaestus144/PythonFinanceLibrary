import unittest

import numpy as np

from src.Finance.Enums.LongShort import LongShort
from src.Finance.ForwardPricing.Forward import Forward


class TestsForwardPricing(unittest.TestCase):
    def test_forward_price(self):
        forward = Forward(100, 110, 0.1, 1, 0.2, LongShort.LONG)
        expected_price = 100 * np.exp(0.1 * 1) - 110
        self.assertAlmostEqual(expected_price, forward.price(), 2)

    def test_monte_carlo_forward_price(self):
        forward = Forward(100, 110, 0.1, 1, 0.2, LongShort.LONG)
        expected_price = 100 * np.exp(0.1 * 1) - 110
        print(expected_price)
        [price, stddev] = forward.monte_carlo_price(10000, 0.5)
        print(price)
        self.assertTrue(expected_price - 2*stddev < price < expected_price + 2*stddev)

from src.Finance.Curves.SurvivalCurve import SurvivalCurve

import numpy as np
import unittest

class SurvivalCurve_Tests(unittest.TestCase):
    # driftless test
    # pfe type test
    # stats tests
    def test_get_default_probability_interpolation(self):
        tenors = np.array([0, 0.25, 0.5])
        default_probabilities = np.array([0, 0.1, 0.2])
        survival_curve = SurvivalCurve(tenors, default_probabilities, 0.4)
        expected_survival_probabilities = np.array([0, 0.05, 0.1, 0.15, 0.2])
        np.testing.assert_array_almost_equal(
            survival_curve.get_probabilities_of_default([0, 0.125, 0.25, 0.375, 0.5]),
            expected_survival_probabilities, 2)

    def test_get_default_probability_extrapolation(self):
        tenors = np.array([0, 0.25, 0.5])
        default_probabilities = np.array([0, 0.1, 0.2])
        survival_curve = SurvivalCurve(tenors, default_probabilities, 0.4)
        expected_survival_probabilities = np.array([0.2])
        np.testing.assert_array_almost_equal(
            survival_curve.get_probabilities_of_default([0.75]),
            expected_survival_probabilities, 2)

    def test_plot_curve(self):
        survival_curve\
            = SurvivalCurve(np.array([0.00, 0.48, 0.99, 1.98, 2.98, 3.98, 4.98, 6.99, 9.99]),
                            np.array([0.000, 0.0160, 0.0228, 0.0507, 0.0456, 0.0454, 0.0462, 0.0878, 0.1144]),
                            recovery_rate=0.4)
        survival_curve.plot_curve()

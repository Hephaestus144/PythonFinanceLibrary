import numpy as np

from src.Finance.Curves.InterestRateCurve import InterestRateCurve
from src.Finance.Curves.RateConvention import RateConvention

import unittest


class TestsInterestRateCurves(unittest.TestCase):
    #region discount factors
    def test_discount_factor_interpolation(self):
        interest_rate_curve \
            = InterestRateCurve(
                tenors=np.array([0.25, 0.5, 0.75, 1.0]),
                discount_factors=np.array([0.9, 0.85, 0.8, 0.75]))
        actual_discount_factors \
            = interest_rate_curve.get_discount_factors(
                np.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]))
        expected_discount_factors = np.array([0.95, 0.9, 0.875, 0.85, 0.825, 0.8, 0.775, 0.75])
        np.testing.assert_array_almost_equal(expected_discount_factors, actual_discount_factors, 2)

    def test_discount_factor_extrapolation(self):
        interest_rate_curve \
            = InterestRateCurve(
                tenors=np.array([0.25, 0.5, 0.75, 1.0]),
                discount_factors=np.array([0.9, 0.85, 0.8, 0.75]))
        actual_discount_factors \
            = interest_rate_curve.get_discount_factors(np.array([2.0]))
        expected_discount_factors = np.array([0.75])
        np.testing.assert_array_almost_equal(expected_discount_factors, actual_discount_factors, 2)

    #endregion discount factors

    #region zero rates
    def test_nacc_zero_rates_without_interpolation(self):
        tenors = np.array([0.25, 0.5, 0.75, 1.0])
        expected_rates = np.array([0.1, 0.11, 0.12, 0.13])
        discount_factors = np.exp(-tenors * expected_rates)
        interest_rate_curve = InterestRateCurve(tenors=tenors, discount_factors=discount_factors)
        actual_rates = interest_rate_curve.get_zero_rates(tenors, RateConvention.NACC)
        np.testing.assert_array_almost_equal(expected_rates, actual_rates, 2)

    def test_nacm_zero_rates_without_interpolation(self):
        tenors = np.array([0.25, 0.5, 0.75, 1.0])
        expected_rates = np.array([0.1, 0.11, 0.12, 0.13])
        discount_factors = (1 + expected_rates / 12) ** (-12 * tenors)
        interest_rate_curve = InterestRateCurve(tenors=tenors, discount_factors=discount_factors)
        actual_rates = interest_rate_curve.get_zero_rates(tenors, RateConvention.NACA)
        np.testing.assert_array_almost_equal(expected_rates, actual_rates, 2)

    def test_nacq_zero_rates_without_interpolation(self):
        tenors = np.array([0.25, 0.5, 0.75, 1.0])
        expected_rates = np.array([0.1, 0.11, 0.12, 0.13])
        discount_factors = (1 + expected_rates / 4) ** (-4 * tenors)
        interest_rate_curve = InterestRateCurve(tenors=tenors, discount_factors=discount_factors)
        actual_rates = interest_rate_curve.get_zero_rates(tenors, RateConvention.NACA)
        np.testing.assert_array_almost_equal(expected_rates, actual_rates, 2)

    def test_nacs_zero_rates_without_interpolation(self):
        tenors = np.array([0.25, 0.5, 0.75, 1.0])
        expected_rates = np.array([0.1, 0.11, 0.12, 0.13])
        discount_factors = (1 + expected_rates / 2) ** (-2 * tenors)
        interest_rate_curve = InterestRateCurve(tenors=tenors, discount_factors=discount_factors)
        actual_rates = interest_rate_curve.get_zero_rates(tenors, RateConvention.NACA)
        np.testing.assert_array_almost_equal(expected_rates, actual_rates, 2)

    def test_naca_zero_rates_without_interpolation(self):
        tenors = np.array([0.25, 0.5, 0.75, 1.0])
        expected_rates = np.array([0.1, 0.11, 0.12, 0.13])
        discount_factors = (1 + expected_rates) ** (-1 * tenors)
        interest_rate_curve = InterestRateCurve(tenors=tenors, discount_factors=discount_factors)
        actual_rates = interest_rate_curve.get_zero_rates(tenors, RateConvention.NACA)
        np.testing.assert_array_almost_equal(expected_rates, actual_rates, 2)

    #endregion zero rates

    #region forward rates
    def test_nacc_forward_rates(self):
        tenors = np.array([0, 0.25, 0.5, 0.75, 1.00])
        rates = np.array([0, 0.1, 0.11, 0.12, 0.13])
        ir_curve = InterestRateCurve(tenors=tenors, rates=rates, rate_convention=RateConvention.NACC)
        start_tenors = np.array([0.25, 0.5])
        end_tenors = np.array([0.5, 0.75])
        actual_forward_rates = ir_curve.get_forward_rates(start_tenors, end_tenors)
        expected_forward_rates = np.array([(0.11*0.5 - 0.1*0.25)/0.25, (0.12*0.75 - 0.11*0.5)/0.25])
        np.testing.assert_array_almost_equal(expected_forward_rates, actual_forward_rates, 2)

    #endregion forward rates

    def test_curve_plot(self):
        tenors = np.array(
            [0.0, 0.01, 0.08, 0.26, 0.51, 0.76, 1.01, 1.26, 1.50, 1.75, 2.01, 3.00,
             4.00, 5.01, 6.01, 7.01, 8.01, 9.01, 10.01, 12.01, 15.01, 20.01, 25.02, 30.02])
        rates = np.array(
            [0.0659, 0.0659, 0.0689, 0.0696, 0.0684, 0.0676, 0.0670, 0.0666, 0.0665, 0.0665, 0.0665, 0.0671,
             0.0686, 0.0701, 0.0719, 0.0736, 0.0753, 0.0769, 0.0782, 0.0806, 0.0825, 0.0830, 0.0804, 0.0773])
        zar_ir_curve = InterestRateCurve(tenors=tenors, rates=rates, rate_convention=RateConvention.NACC)
        zar_ir_curve.plot_curve('rates')

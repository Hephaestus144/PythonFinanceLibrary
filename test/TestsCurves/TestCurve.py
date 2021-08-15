import numpy as np
from src.Curves.Curve import Curve


def test_get_discount_factors_at_nodes_points():
    curve: Curve = Curve(tenors=[0.25, 0.50, 0.75, 1.00], discount_factors=[0.95, 0.90, 0.85, 0.80])
    expected: np.ndarray = np.array([0.95, 0.90, 0.85, 0.80])
    actual: np.ndarray = curve.get_discount_factors(np.array([0.25, 0.50, 0.75, 1.00]))
    assert np.allclose(expected, actual)


def test_get_discount_factor_at_time_zero():
    curve: Curve = Curve(tenors=[0.25, 0.50, 0.75, 1.00], discount_factors=[0.95, 0.90, 0.85, 0.80])
    assert curve.get_discount_factors(np.array([0.0]))[0] == 1.0


def test_curve_constructor_from_discount_factors():
    curve = Curve(tenors=[0.25, 0.50, 0.75, 1.00], discount_factors=[0.95, 0.90, 0.85, 0.80])
    expected: np.ndarray = -1 * np.log([0.95, 0.90, 0.85, 0.80]) / np.array([0.25, 0.50, 0.75, 1.00])
    assert np.array_equiv(expected, curve.zero_rates)


def test_curve_constructor_from_zero_rates():
    curve = Curve(tenors=[0.25, 0.50, 0.75, 1.00], zero_rates=[0.05, 0.10, 0.15, 0.20])
    expected: np.ndarray = np.exp(-1 * np.array([0.05, 0.10, 0.15, 0.20]) * np.array([0.25, 0.50, 0.75, 1.00]))
    expected = np.insert(expected, 0, 1)
    assert np.array_equiv(expected, curve.discount_factors)


def test_discount_factor_at_time_zero():
    curve = Curve(tenors=[0.25, 0.50, 0.75, 1.00], discount_factors=[0.95, 0.90, 0.85, 0.80])
    assert curve.get_discount_factors(np.array(0)) == 1


def test_multi_point_discount_factors():
    curve = Curve(tenors=[0.25, 0.50, 0.75, 1.00], discount_factors=[0.95, 0.90, 0.85, 0.80])
    assert np.array_equiv(
        curve.get_discount_factors(np.array([0.25, 0.375, 0.5, 0.625])),
        np.array([0.95, 0.925, 0.90, 0.875]))


def test_discount_factor_right_extrapolation():
    curve = Curve(tenors=[0.25, 0.50, 0.75, 1.00], discount_factors=[0.95, 0.90, 0.85, 0.80])
    assert curve.get_discount_factors(np.array(1.25)) == 0.80


def test_discount_factor_left_extrapolation():
    curve = Curve(tenors=[0.25, 0.50, 0.75, 1.00], discount_factors=[0.95, 0.90, 0.85, 0.80])
    assert curve.get_discount_factors(np.array(0.125)) == 0.975


def test_forward_rates():
    curve = Curve(tenors=[0.25, 0.50, 0.75, 1.00], discount_factors=[0.95, 0.90, 0.85, 0.80])
    start_points: np.ndarray = np.array([0.00, 0.25, 0.50, 0.75])
    end_points: np.ndarray = np.array([0.25, 0.50, 0.75, 1.00])
    expected_forward_rates: np.ndarray = np.array([0.20517318, 0.21626889, 0.22863366, 0.24249849])
    actual_forward_rates: np.ndarray = curve.get_forward_rates(start_points, end_points)
    assert np.allclose(expected_forward_rates, actual_forward_rates)


def test_get_forward_discount_factors():
    curve = Curve(tenors=[0.25, 0.50, 0.75, 1.00], discount_factors=[0.95, 0.90, 0.85, 0.80])
    actual = curve.get_forward_discount_factors(0.25, np.array([0.25, 0.50, 0.75, 1.00]))
    expected = np.append(1, np.array([0.90, 0.85, 0.80]) / 0.95)
    assert np.allclose(actual, expected)


def test_get_zero_rates_flat_curve_single_point():
    tenors: np.ndarray = np.array([1.0])
    zero_rates: np.ndarray = np.array([0.1])
    curve: Curve = Curve(tenors=tenors, zero_rates=zero_rates)
    assert np.allclose(curve.get_zero_rates(tenors), zero_rates)


def test_get_zero_rates_flat_curve_multiple_points():
    curve_tenors: np.ndarray = np.array([1.0])
    zero_rates: np.ndarray = np.array([0.1])
    curve: Curve = Curve(tenors=curve_tenors, zero_rates=zero_rates)
    test_tenors: np.ndarray = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    assert np.allclose(np.array([0.1, 0.1, 0.1, 0.1, 0.1]), curve.get_zero_rates(test_tenors))

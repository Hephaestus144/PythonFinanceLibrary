import numpy as np
from src.Curves.Curve import Curve


def test_discount_factor_at_time_zero():
    curve = Curve(tenors=[0.25, 0.50, 0.75, 1.00], discount_factors=[0.95, 0.90, 0.85, 0.80])
    assert curve.get_discount_factors(0) == 1


def test_multi_point_discount_factors():
    curve = Curve(tenors=[0.25, 0.50, 0.75, 1.00], discount_factors=[0.95, 0.90, 0.85, 0.80])
    assert np.array_equiv(
        curve.get_discount_factors(np.array([0.25, 0.375, 0.5, 0.625])),
        np.array([0.95, 0.925, 0.90, 0.875]))


def test_discount_factor_right_extrapolation():
    curve = Curve(tenors=[0.25, 0.50, 0.75, 1.00], discount_factors=[0.95, 0.90, 0.85, 0.80])
    assert curve.get_discount_factors(1.25) == 0.80


def test_discount_factor_left_extrapolation():
    curve = Curve(tenors=[0.25, 0.50, 0.75, 1.00], discount_factors=[0.95, 0.90, 0.85, 0.80])
    assert curve.get_discount_factors(0.125) == 0.975

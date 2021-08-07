import numpy as np
import pytest

from src.Curves.Curve import Curve
from src.StochasticProcesses.HullWhite import HullWhite


@pytest.fixture
def hw_constant_sigma():
    tenors = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
    discount_factors = np.array([1.000000000, 0.975309912, 0.951229425, 0.927743486, 0.904837418])
    curve = Curve(tenors=tenors, discount_factors=discount_factors)
    alpha = 0.1
    return HullWhite(alpha, None, [0.8], curve)


@pytest.fixture
def hw_variable_sigma():
    tenors = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
    discount_factors = np.array([1.000000000, 0.975309912, 0.951229425, 0.927743486, 0.904837418])
    curve = Curve(tenors=tenors, discount_factors=discount_factors)
    alpha = 0.1
    sigma_tenors = np.array([0.00, 0.5])
    sigmas = np.array([0.2, 0.15])
    return HullWhite(alpha, sigma_tenors, sigmas, curve)


def test_b_function(hw_constant_sigma):
    alpha = 0.1
    actual = hw_constant_sigma.b_function(0, 1)
    expected = (1/alpha)*(1 - np.exp(-1 * alpha))
    assert actual == expected


@pytest.mark.parametrize("time,expected", [(0.25, 0.20), (0.75, 0.20)])
def test_interpolate_constant_sigma(hw_constant_sigma, time, expected):
    assert hw_constant_sigma.interpolate_sigma(time) == expected


@pytest.mark.parametrize("time,expected", [(0.25, 0.20), (0.75, 0.15)])
def test_interpolate_variable_sigma(hw_variable_sigma, time, expected):
    assert hw_variable_sigma.interpolate_sigma(time) == expected


def test_swaption_pricing_vol(hw_constant_sigma):
    swap_cashflow_tenors = np.array([0.25, 0.50, 0.75, 1.00])
    actual = hw_constant_sigma.swaption_pricing_vol(time=0.0, strike=0.1, swap_cashflow_tenors=swap_cashflow_tenors)
    print(f'Actual: {np.sqrt(actual)}')


def test_swaption_pricer(hw_constant_sigma):
    swap_cashflow_tenors = np.array([0.25, 0.50, 0.75, 1.00])
    actual = hw_constant_sigma.swaption_pricer(0.1, swap_cashflow_tenors)
    print(f'Actual: {actual}\n')

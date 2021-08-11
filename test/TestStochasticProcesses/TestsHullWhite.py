import matplotlib.pyplot as plt
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
    return HullWhite(alpha, None, [0.2], curve)


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
    swap_cashflow_tenors = np.array([1.25, 1.50, 1.75, 2.00])
    actual =\
        hw_constant_sigma.swaption_pricing_vol(
            time=0.5,
            strike=0.1,
            swaption_expiry=1.0,
            swap_cashflow_tenors=swap_cashflow_tenors)

    print(f'\nActual: {np.sqrt(actual)}\n')


def test_weighted_strike(hw_constant_sigma):
    swap_cashflow_tenors = np.array([1.25, 1.50, 1.75, 2.00])
    h0 = hw_constant_sigma.weighted_strike(strike=0.1, swaption_expiry=1.0, swap_cashflow_tenors=swap_cashflow_tenors)
    print(f'\n{h0=}\n')


def test_swaption_pricer():
    tenors = np.array([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50])
    discount_factors = np.array([1.000000000, 0.975309912, 0.951229425, 0.927743486, 0.904837418, 0.882496903,
                                 0.860707976, 0.839457021, 0.818730753, 0.798516219, 0.778800783])
    curve = Curve(tenors=tenors, discount_factors=discount_factors)
    alpha = 0.01
    hw = HullWhite(alpha, None, [0.5], curve)

    swap_cashflow_tenors = np.array([1.25, 1.50, 1.75, 2.00])
    actual =\
        hw.swaption_price(
            strike=0.1,
            swaption_expiry=1.0,
            swap_cashflow_tenors=swap_cashflow_tenors)

    print(f'Actual: {actual}\n')


def test_plot_swaption_price():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})
    curve_tenors = np.array([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50])
    discount_factors = np.array([1.000000000, 0.975309912, 0.951229425, 0.927743486, 0.904837418, 0.882496903,
                                 0.860707976, 0.839457021, 0.818730753, 0.798516219, 0.778800783])
    curve = Curve(tenors=curve_tenors, discount_factors=discount_factors)

    alphas = np.arange(0.05, 0.5, 0.05)
    sigmas = np.arange(0.0, 2.0, 0.1)
    prices = np.zeros([len(alphas), len(sigmas)])
    for i in range(0, len(alphas)):
        for j in range(0, len(sigmas)):
            hw = HullWhite(alphas[i], None, [sigmas[j]], curve)
            swap_cashflow_tenors = np.array([1.25, 1.50, 1.75, 2.00])
            prices[i, j] =\
                hw.swaption_price(
                    strike=0.1,
                    swaption_expiry=1.0,
                    swap_cashflow_tenors=swap_cashflow_tenors)

    X, Y = np.meshgrid(sigmas, alphas)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    surf = ax[0].plot_surface(X, Y, prices, cmap=plt.get_cmap('gnuplot'))
    ax[0].set_xlabel(r'$\sigma$', size=20)
    ax[0].set_ylabel(r'$\alpha$', size=20)
    ax[0].set_zlabel('Swaption Price', size=15)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax[0].view_init(30, 135)
    ax[0].set_title(r'Swaption Price as a Function of $\alpha$ and $\sigma$')
    plt.show()


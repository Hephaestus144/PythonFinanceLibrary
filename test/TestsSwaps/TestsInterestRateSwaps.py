import numpy as np
import pytest

from src.Curves.Curve import Curve
from src.Enums.Frequency import Frequency
from src.Swaps.InterestRateSwap import InterestRateSwap


@pytest.fixture
def irs():
    return InterestRateSwap(1_000_000, 0.1, 0.0, 1.0, Frequency.QUARTERLY)


@pytest.fixture
def flat_curve():
    return Curve(tenors=np.array([1.0]), zero_rates=np.array([0.1]))


@pytest.fixture
def increasing_curve():
    return Curve(
        tenors=np.array([0.25, 0.50, 0.75, 1.00]),
        zero_rates=np.array([0.10, 0.11, 0.12, 0.13]))


def test_payment_tenors(irs):
    expected = np.linspace(start=0.25, stop=1.0, num=int(1.00 / 0.25), endpoint=True)
    assert np.allclose(expected, irs.payment_tenors)


def test_compute_swap_fair_rate_flat_curve(irs, flat_curve):
    assert np.allclose(irs.compute_fair_swap_rate(flat_curve), 0.1)


def test_compute_swap_fair_rate_increasing_curve(irs, increasing_curve):
    assert np.allclose(irs.compute_fair_swap_rate(increasing_curve), 0.1291270517696489)


def test_day_count_fractions(irs):
    assert np.allclose(np.array([0.25, 0.25, 0.25, 0.25]), irs.day_count_fractions)

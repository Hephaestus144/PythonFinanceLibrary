import numpy as np

from src.Curves.Curve import Curve
from src.Swaps.Frequency import Frequency
from src.Swaps.InterestRateSwap import InterestRateSwap


def test_payment_tenors():
    curve = Curve(tenors=np.array([1.00]), zero_rates=np.array([0.1]))
    irs = InterestRateSwap(1000000, 0.1, 1.0, Frequency.QUARTERLY)
    expected = np.arange(0.25, 1.0, 0.25)
    assert np.allclose(irs.payment_tenors, expected)

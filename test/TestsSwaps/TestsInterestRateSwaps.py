import numpy as np

from src.Curves.Curve import Curve
from src.Swaps.Frequency import Frequency
from src.Swaps.InterestRateSwap import InterestRateSwap


def test_exchange_tenors():
    curve = Curve(tenors=np.array([1.00]), zero_rates=np.array([0.1]))
    irs = InterestRateSwap(1000000, 0.1, 1.0, Frequency.QUARTERLY)
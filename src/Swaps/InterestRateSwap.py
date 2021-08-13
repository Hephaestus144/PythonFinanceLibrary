from src.Curves.Curve import Curve
from src.Swaps.Frequency import Frequency


class InterestRateSwap:
    """
    This is a vanilla interest rate swap i.e. it cannot handle basis.
    """
    def __init__(self, notional: float, strike: float, tenor: float, payment_frequency: Frequency, curve: Curve):
        self.notional: float = notional
        self.strike: float = strike
        self.tenor: float = tenor
        self.payment_frequency: Frequency = payment_frequency
        self.curve: Curve = curve



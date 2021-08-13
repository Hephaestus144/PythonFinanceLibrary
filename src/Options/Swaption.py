from src.Curves.Curve import Curve
from src.Options.PayerReceiver import PayerReceiver
from src.StochasticProcesses.GBM import GBM


# TODO: Make it more generic than just a quarterly swap.
class Swaption:
    def __init__(self,
                 strike: float,
                 curve: Curve,
                 volatility: float,
                 swaption_maturity_tenor: float,
                 swap_tenor: float,
                 payer_receiver: PayerReceiver):
        self.strike: float = strike
        self.curve: Curve = curve
        self.volatility: float = volatility
        self.swaption_maturity_tenor: float = swaption_maturity_tenor
        self.swap_tenor: float = swap_tenor
        self.payer_receiver: PayerReceiver = payer_receiver

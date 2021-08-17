from scipy.stats import norm
from src.Curves.Curve import Curve
from src.Options.PayerReceiver import PayerReceiver
from src.Swaps.Frequency import Frequency
from src.Swaps.InterestRateSwap import InterestRateSwap


import numpy as np


class Swaption:
    def __init__(self,
                 notional: float,
                 strike: float,
                 swaption_expiry_tenor: float,
                 swap_start_tenor: float,
                 swap_end_tenor: float,
                 swap_payment_frequency: Frequency,
                 payer_receiver: PayerReceiver):
        self.notional: float = notional
        self.strike: float = strike
        self.swaption_expiry_tenor: float = swaption_expiry_tenor
        self.swap_start_tenor: float = swap_start_tenor
        self.swap_end_tenor: float = swap_end_tenor
        self.swap_payment_frequency = swap_payment_frequency
        self.payer_receiver: PayerReceiver = payer_receiver
        self.irs: InterestRateSwap = \
            InterestRateSwap(
                self.notional,
                self.strike,
                self.swap_start_tenor,
                self.swap_end_tenor,
                self.swap_payment_frequency)

    def __blacks_formula(self, curve: Curve, vol: float):
        direction: float = 1.0 if self.payer_receiver == PayerReceiver.PAYER else -1.0
        s: float = self.irs.compute_fair_swap_rate(curve)
        d1: float =\
            (np.log(s / self.strike) + 0.5 * vol ** 2 * self.swaption_expiry_tenor) / \
            vol * np.sqrt(self.swaption_expiry_tenor)
        d2: float =\
            (np.log(s / self.strike) - 0.5 * vol**2 * self.swaption_expiry_tenor) / \
            vol * np.sqrt(self.swaption_expiry_tenor)
        return direction * (s * norm.cdf(direction * d1) - self.strike * norm.cdf(direction * d2))

    def black_price(self, curve: Curve, vol: float):
        dfs: np.ndarray = curve.get_discount_factors(self.irs.payment_tenors)
        annuity_factor: float = 0
        for i, df in enumerate(dfs):
            annuity_factor += self.irs.day_count_fractions[i] * df

        return self.notional * self.__blacks_formula(curve, vol) * annuity_factor

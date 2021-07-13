import numpy as np

from src.Finance.Curves.InterestRateCurve import InterestRateCurve
from src.Finance.Curves.SurvivalCurve import SurvivalCurve


class CVA:
    def __init__(self, interest_rate_curve: InterestRateCurve, survival_curve: SurvivalCurve):
        self.interest_rate_curve = interest_rate_curve
        self.survival_curve = survival_curve

    def compute_cva(self, tenors, price_paths):
        discount_factors = self.interest_rate_curve.get_discount_factors(tenors)
        default_probabilities = self.survival_curve.get_probabilities_of_default(tenors)
        return np.average(np.sum((1 - self.survival_curve.recovery_rate)
                                 * np.maximum(price_paths, 0) * discount_factors
                                 * default_probabilities, 1))

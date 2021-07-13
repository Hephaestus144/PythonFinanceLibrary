import numpy as np
import unittest

from src.Finance.XVA.CVA import CVA
from src.Finance.OptionPricing.EuropeanOption import EuropeanOption
from src.Finance.OptionPricing.OptionStyle import OptionStyle
from src.Finance.Curves.InterestRateCurve import InterestRateCurve
from src.Finance.Curves.RateConvention import RateConvention
from src.Finance.Curves.SurvivalCurve import SurvivalCurve


class TestsCVA(unittest.TestCase):
    # the CVA on an option & a forward should be the same
    def test_option_and_forward_cva(self):
        fxSpot = 1/(14.139235*1.16)
        strike = 1/11.00
        r = -0.04
        #fx1YVol = 0.14
        fx1YVol = 0.173 #1545
        fx2YVol = 0.1675
        T = 4.23
        #T = 1.5
        european_option\
            = EuropeanOption(fxSpot, strike, r, fx1YVol, T, OptionStyle.PUT)
        [tenors, price_paths] = european_option.monte_carlo_price(100000, 0.2, True)

        rates_tenors = np.array(
            [0.0, 0.01, 0.08, 0.26, 0.51, 0.76, 1.01, 1.26, 1.50, 1.75, 2.01, 3.00,
             4.00, 5.01, 6.01, 7.01, 8.01, 9.01, 10.01, 12.01, 15.01, 20.01, 25.02, 30.02])

        zar_rates = np.array(
            [0.0659, 0.0659, 0.0689, 0.0696, 0.0684, 0.0676, 0.0670, 0.0666, 0.0665, 0.0665, 0.0665, 0.0671,
             0.0686, 0.0701, 0.0719, 0.0736, 0.0753, 0.0769, 0.0782, 0.0806, 0.0825, 0.0830, 0.0804, 0.0773])
        zar_ir_curve = InterestRateCurve(tenors=rates_tenors, rates=zar_rates, rate_convention=RateConvention.NACC)

        zar_usd_basis_tenors = np.array([0.00, 0.02, 0.04, 0.06, 0.08, 0.17, 0.25, 0.34, 0.42, 0.50,
                0.75, 1.00, 2.00, 3.01, 4.01, 5.01, 6.01, 7.01, 8.01, 9.01, 10.01])

        zar_usd_basis_rates = np.array([0.0678, 0.0698, 0.0694, 0.0688, 0.0685, 0.0672, 0.0665, 0.0661,
                                        0.0656, 0.0650, 0.0644, 0.0639, 0.0643, 0.0651, 0.0670, 0.0688,
                                        0.0710, 0.0728, 0.0747, 0.0764, 0.0777])
        zar_ir_basis_curve = InterestRateCurve(tenors=zar_usd_basis_tenors, rates=zar_usd_basis_rates, rate_convention=RateConvention.NACC)

        survival_curve\
            = SurvivalCurve(tenors=np.array([0.00, 0.48, 0.99, 1.98, 2.98, 3.98, 4.98, 6.99, 9.99]),
                            probabilities_of_default=np.array([1.0000, 0.9840, 0.9612, 0.9106, 0.8650, 0.8196, 0.7733, 0.6855, 0.5711]),
                            recovery_rate=0.0)

        cva = CVA(zar_ir_basis_curve, survival_curve)
        print(f'{11000000000 * cva.compute_cva(tenors, price_paths)*0.2:,}')


    # CVA on something that is more positive should be higher
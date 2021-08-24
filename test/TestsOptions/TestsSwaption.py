import numpy as np
import pytest

from src.Curves.Curve import Curve
from src.Enums.Frequency import Frequency
from src.Options.Swaption import PayerReceiver
from src.Options.Swaption import Swaption


@pytest.fixture
def flat_curve():
    return Curve(tenors=np.array([1.0]), zero_rates=np.array([0.1]))


def test_swaption_irs(flat_curve):
    swaption: Swaption = Swaption(1.0, 0.1, 1.0, 1.0, 2.0, Frequency.QUARTERLY, PayerReceiver.PAYER)
    assert np.allclose(0.023270738579718328, swaption.black_price(flat_curve, vol=0.7))
